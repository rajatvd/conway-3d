"""Reproducible tabletop Life sculptures with registered time-band colors.

Usage: python sculptures.py OUTPUT --font /path/to/CascadiaMono.ttf
No slicing, printer integration, or network calls are performed.
"""

import argparse
import itertools
import json
import shutil
import time
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import trimesh
from lib3mf import get_wrapper

from conway3d import boards_to_mesh
from conway_to_stl import conway_to_boards
from sculpture_preview import checked_jpeg, overview, render

CORE = "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"
ROOT = Path(__file__).resolve().parent
# Rotate Z-up millimeters to glTF's right-handed Y-up meters.
TO_GLTF = np.array(
    [[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=float
)


def mesh_stats(mesh):
    return {
        "dimensions_mm": mesh.extents.tolist(),
        "bounds_mm": mesh.bounds.tolist(),
        "vertices": len(mesh.vertices),
        "triangles": len(mesh.faces),
        "finite_vertices": bool(np.isfinite(mesh.vertices).all()),
        "watertight": bool(mesh.is_watertight),
        "consistent_winding": bool(mesh.is_winding_consistent),
        "positive_volume": bool(mesh.is_volume),
        "volume_mm3": float(mesh.volume),
        "components": len(mesh.split(only_watertight=False)),
    }


def integrated_base(mesh, height, padding):
    """Octagonal plinth, including 0.2 mm overlap with the unaltered links."""
    lo, hi = mesh.bounds[:, :2]
    lo, hi = lo - padding, hi + padding
    corner = min(3.0, float(min(hi - lo) / 6))
    ring = np.array(
        [
            [lo[0] + corner, lo[1]],
            [hi[0] - corner, lo[1]],
            [hi[0], lo[1] + corner],
            [hi[0], hi[1] - corner],
            [hi[0] - corner, hi[1]],
            [lo[0] + corner, hi[1]],
            [lo[0], hi[1] - corner],
            [lo[0], lo[1] + corner],
        ]
    )
    center = (lo + hi) / 2
    inset = (ring - center) * ((hi - lo - 0.8) / (hi - lo)) + center
    vertices = np.vstack(
        [
            np.column_stack([ring, np.zeros(8)]),
            np.column_stack([ring, np.full(8, height - 0.6)]),
            np.column_stack([inset, np.full(8, height + 0.2)]),
        ]
    )
    return trimesh.convex.convex_hull(vertices)


def split_epochs(mesh, cuts_mm):
    """Intersect ONE complete union with touching closed Z slabs."""
    lo, hi = mesh.bounds
    bounds = [lo[2] - 1, *cuts_mm, hi[2] + 1]
    parts = []
    for bottom, top in itertools.pairwise(bounds):
        box = trimesh.creation.box(
            extents=[hi[0] - lo[0] + 2, hi[1] - lo[1] + 2, top - bottom]
        )
        box.apply_translation(
            [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (bottom + top) / 2]
        )
        parts.append(trimesh.boolean.intersection([mesh, box], engine="manifold"))
    return parts


def export_3mf(parts, palette, path):
    """Core 3MF: one assembly, three base-material mesh components, mm."""
    ET.register_namespace("", CORE)

    def child(parent, tag, **attrs):
        return ET.SubElement(
            parent, f"{{{CORE}}}{tag}", {k: str(v) for k, v in attrs.items()}
        )

    model = ET.Element(
        f"{{{CORE}}}model",
        {"unit": "millimeter", "{http://www.w3.org/XML/1998/namespace}lang": "en-US"},
    )
    resources = child(model, "resources")
    materials = child(resources, "basematerials", id=1)
    for color in palette:
        child(materials, "base", name=color["name"], displaycolor=color["hex"] + "FF")
    for i, (mesh, color) in enumerate(zip(parts, palette)):
        obj = child(
            resources,
            "object",
            id=i + 2,
            type="model",
            pid=1,
            pindex=i,
            name=color["name"],
        )
        body = child(obj, "mesh")
        vertices = child(body, "vertices")
        for x, y, z in mesh.vertices:
            child(vertices, "vertex", x=f"{x:.9g}", y=f"{y:.9g}", z=f"{z:.9g}")
        triangles = child(body, "triangles")
        for a, b, c in mesh.faces:
            child(triangles, "triangle", v1=a, v2=b, v3=c)
    assembly = child(resources, "object", id=5, type="model", name="Time-band assembly")
    components = child(assembly, "components")
    for i in range(len(parts)):
        child(components, "component", objectid=i + 2)
    child(child(model, "build"), "item", objectid=5)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "[Content_Types].xml",
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>'
            "</Types>",
        )
        archive.writestr(
            "_rels/.rels",
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Target="/3D/3dmodel.model" Id="rel0" '
            'Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>'
            "</Relationships>",
        )
        archive.writestr(
            "3D/3dmodel.model",
            ET.tostring(model, encoding="utf-8", xml_declaration=True),
        )


def export_glb(parts, palette, path):
    scene = trimesh.Scene()
    for mesh, color in zip(parts, palette):
        mesh = mesh.copy()
        # glTF linear coordinates are meters; STL and 3MF remain millimeters.
        mesh.apply_transform(TO_GLTF)
        mesh.apply_scale(0.001)
        rgba = [*bytes.fromhex(color["hex"][1:]), 255]
        material = trimesh.visual.material.PBRMaterial(
            name=color["name"],
            baseColorFactor=rgba,
            metallicFactor=0,
            roughnessFactor=0.68,
        )
        mesh.visual = trimesh.visual.TextureVisuals(material=material)
        scene.add_geometry(mesh, node_name=color["name"], geom_name=color["name"])
    scene.export(path)


def validate_assets(directory):
    """Independently reload disk artifacts, including XML materials/assembly."""
    directory = Path(directory)
    complete = trimesh.load_mesh(directory / "complete.stl", process=True)
    config = json.loads((directory / "settings.json").read_text())
    parts = [
        trimesh.load_mesh(
            directory / f"{i + 1:02}-{c['name'].lower()}.stl", process=True
        )
        for i, c in enumerate(config["palette"])
    ]
    stats = mesh_stats(complete)
    part_stats = [mesh_stats(p) for p in parts]
    for item in [stats, *part_stats]:
        assert all(
            item[key]
            for key in [
                "finite_vertices",
                "watertight",
                "consistent_winding",
                "positive_volume",
            ]
        ), item
    assert stats["components"] == 1, stats
    rebuilt = trimesh.boolean.union(parts, engine="manifold")
    added = trimesh.boolean.difference([rebuilt, complete], engine="manifold").volume
    missing = trimesh.boolean.difference([complete, rebuilt], engine="manifold").volume
    overlap = sum(p.volume for p in parts) - rebuilt.volume
    tolerance = max(0.01, complete.volume * 1e-6)
    assert max(abs(added), abs(missing), abs(overlap)) < tolerance, (
        added,
        missing,
        overlap,
    )
    assert np.allclose(rebuilt.bounds, complete.bounds, atol=2e-5)
    # Decode the delivered XML ourselves; do not trust our writer's in-memory meshes.
    with zipfile.ZipFile(directory / "multicolor.3mf") as archive:
        assert archive.testzip() is None
        xml = ET.fromstring(archive.read("3D/3dmodel.model"))
    ns = {"m": CORE}
    assert xml.attrib["unit"] == "millimeter"
    resources = xml.find("m:resources", ns)
    material_list = resources.find("m:basematerials", ns)
    assert [b.attrib["displaycolor"] for b in material_list] == [
        c["hex"] + "FF" for c in config["palette"]
    ]
    mesh_objects = resources.findall("m:object[m:mesh]", ns)
    assert len(mesh_objects) == len(parts)
    for index, (obj, part) in enumerate(zip(mesh_objects, parts)):
        assert obj.attrib["pid"] == "1" and int(obj.attrib["pindex"]) == index
        points = [
            [float(v.attrib[k]) for k in ("x", "y", "z")]
            for v in obj.findall("m:mesh/m:vertices/m:vertex", ns)
        ]
        faces = [
            [int(v.attrib[k]) for k in ("v1", "v2", "v3")]
            for v in obj.findall("m:mesh/m:triangles/m:triangle", ns)
        ]
        decoded = trimesh.Trimesh(points, faces, process=True)
        assert decoded.is_volume
        assert np.allclose(decoded.bounds, part.bounds, atol=2e-5)
        # Same vertex/face buffers as the registered STL, not merely a matching bbox.
        assert np.allclose(np.array(points), part.vertices, atol=2e-5)
        assert np.array_equal(faces, part.faces)
    components = resources.find('m:object[@id="5"]/m:components', ns)
    assert [c.attrib["objectid"] for c in components] == ["2", "3", "4"]
    assert all("transform" not in c.attrib for c in components)
    assert xml.find("m:build/m:item", ns).attrib["objectid"] == "5"
    glb = trimesh.load(directory / "multicolor.glb", force="scene")
    assert len(glb.geometry) == 3
    glb.apply_transform(TO_GLTF.T)
    glb.apply_scale(1000)
    assert np.allclose(glb.bounds, complete.bounds, atol=2e-5)
    for color in config["palette"]:
        matches = [
            (key, m)
            for key, m in glb.geometry.items()
            if m.visual.material.name == color["name"]
        ]
        assert len(matches) == 1
        key, geometry = matches[0]
        assert np.array_equal(
            geometry.visual.material.baseColorFactor,
            [*bytes.fromhex(color["hex"][1:]), 255],
        )
        part = parts[config["palette"].index(color)]
        # Scene transforms are evaluated independently of the geometry buffers.
        node = next(n for n in glb.graph.nodes_geometry if glb.graph[n][1] == key)
        transform, _ = glb.graph[node]
        recovered = geometry.copy()
        recovered.apply_transform(transform)
        assert np.allclose(recovered.bounds, part.bounds, atol=2e-5)
        assert np.allclose(recovered.vertices, part.vertices, atol=2e-5)
        assert np.array_equal(recovered.faces, part.faces)
    # A second implementation checks the package in strict standard-reader mode.
    wrapper = get_wrapper()
    model = wrapper.CreateModel()
    reader = model.QueryReader("3mf")
    reader.SetStrictModeActive(True)
    reader.ReadFromFile(str(directory / "multicolor.3mf"))
    assert reader.GetWarningCount() == 0
    assert model.GetMeshObjects().Count() == 3
    assert model.GetComponentsObjects().Count() == 1
    assert model.GetBuildItems().Count() == 1
    iterator = model.GetMeshObjects()
    for part, color in zip(parts, config["palette"]):
        assert iterator.MoveNext()
        obj = iterator.GetCurrentMeshObject()
        assert obj.IsValid() and obj.IsManifoldAndOriented()
        assert obj.GetVertexCount() == len(part.vertices)
        assert obj.GetTriangleCount() == len(part.faces)
        resource, prop, assigned = obj.GetObjectLevelProperty()
        assert assigned
        group = model.GetBaseMaterialGroupByID(resource)
        assert group.GetName(prop) == color["name"]
        rgba = group.GetDisplayColor(prop)
        assert [rgba.Red, rgba.Green, rgba.Blue, rgba.Alpha] == [
            *bytes.fromhex(color["hex"][1:]),
            255,
        ]
    return {
        "lib3mf_strict": {
            "version": list(wrapper.GetLibraryVersion()),
            "warnings": 0,
            "mesh_objects": 3,
            "assemblies": 1,
            "build_items": 1,
            "all_manifold_oriented_and_materials_verified": True,
        },
        "complete": stats,
        "color_parts": part_stats,
        "color_union_added_mm3": float(added),
        "color_union_missing_mm3": float(missing),
        "color_overlap_mm3": float(overlap),
        "equivalence_tolerance_mm3": tolerance,
        "three_mf_registered_materials_and_buffers": True,
        "glb_y_up_meters_colors_and_buffers": True,
    }


def build_design(config, palette, seed_dir, output):
    destination = output / config["id"]
    destination.mkdir(parents=True, exist_ok=False)
    seed = seed_dir / config["seed"]
    shutil.copy2(seed, destination / "seed.txt")
    boards = conway_to_boards(
        seed, config["generations"], world_padding=config["world_padding"]
    )[0]
    # Life uses periodic np.roll. Prove these finite runs never approach its seam.
    for b in boards:
        assert not (b[:2].any() or b[-2:].any() or b[:, :2].any() or b[:, -2:].any())
    mesh = boards_to_mesh(
        boards, angle=config["angle_degrees"], scale=config["cell_mm"]
    )
    mesh.apply_translation(
        [
            -mesh.bounds[:, 0].mean(),
            -mesh.bounds[:, 1].mean(),
            config["base_height_mm"] - mesh.bounds[0, 2],
        ]
    )
    mesh = trimesh.boolean.union(
        [
            mesh,
            integrated_base(mesh, config["base_height_mm"], config["base_padding_mm"]),
        ],
        engine="manifold",
    )
    mesh.export(destination / "complete.stl")
    step_mm = (
        config["cell_mm"] * np.tan(np.radians(config["angle_degrees"])) * np.sqrt(2)
    )
    cuts = [config["base_height_mm"] + t * step_mm for t in config["epoch_cuts"]]
    # Derive color parts from the disk STL, so nothing follows an independent geometry path.
    complete = trimesh.load_mesh(destination / "complete.stl", process=True)
    parts = split_epochs(complete, cuts)
    for i, (part, color) in enumerate(zip(parts, palette)):
        part.export(destination / f"{i + 1:02}-{color['name'].lower()}.stl")
    parts = [
        trimesh.load_mesh(
            destination / f"{i + 1:02}-{c['name'].lower()}.stl", process=True
        )
        for i, c in enumerate(palette)
    ]
    export_3mf(parts, palette, destination / "multicolor.3mf")
    export_glb(parts, palette, destination / "multicolor.glb")
    settings = {
        **config,
        "palette": palette,
        "generation_height_mm": step_mm,
        "epoch_cut_z_mm": cuts,
        "population_by_generation": [int(b.sum()) for b in boards],
        "boundary": "periodic simulation; all live cells stay >1 cell from seam",
        "geometry": "original thin trimesh transitions; manifold union; integrated plinth",
        "finite_cut": f"Source t=0 through t={config['generations']}, not an extinction claim",
    }
    (destination / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    return destination, parts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "seeds/sculptures/designs.json"
    )
    parser.add_argument("--font", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    configs = json.loads(args.config.read_text())
    if not args.validate_only and (args.font is None or not args.font.is_file()):
        parser.error("--font must point to CascadiaMono.ttf")
    args.output.mkdir(parents=True, exist_ok=True)
    results = {}
    previews = []
    for design in configs["designs"]:
        start = time.monotonic()
        if args.validate_only:
            destination = args.output / design["id"]
        else:
            destination, parts = build_design(
                design, configs["palette"], args.config.parent, args.output
            )
            palette = [c["hex"] for c in configs["palette"]]
            subtitle = f"t 0-{design['generations']} / {design['cell_mm']:g} mm cells"
            for suffix, az, el in [
                ("preview", design["view_azimuth"], 24),
                ("detail", design["view_azimuth"] + 95, 38),
            ]:
                render(
                    parts,
                    palette,
                    destination / f"{suffix}.jpg",
                    design["title"],
                    subtitle,
                    args.font,
                    azimuth=az,
                    elevation=el,
                )
        proof = validate_assets(destination)
        proof["images"] = {
            name: checked_jpeg(destination / f"{name}.jpg")
            for name in ["preview", "detail"]
        }
        proof["duration_seconds"] = round(time.monotonic() - start, 3)
        results[design["id"]] = proof
        (destination / "validation.json").write_text(json.dumps(proof, indent=2) + "\n")
        previews.append(destination / "preview.jpg")
        print(design["id"], json.dumps(proof), flush=True)
    if not args.validate_only:
        overview(
            previews,
            args.output / "overview.jpg",
            ["GLIDER / COMET", "PULSAR / CATHEDRAL", "R / BLOOM"],
            args.font,
        )
    results["overview"] = checked_jpeg(args.output / "overview.jpg")
    (args.output / "validation.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
