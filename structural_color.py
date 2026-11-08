"""Volumetric material ownership from real Life birth-link construction.

The input complete mesh is the authority. Envelopes are classification masks,
never replacement geometry. Source-neighbor choices are construction heuristics,
not biological ancestry. Double precision and ASCII STL preserve fine interfaces.
"""

import io
import itertools

import manifold3d as md
import numpy as np
import trimesh

from conway3d import boards_to_mesh, transition_vertices
from trimesh_conway import mesh64, native


def decoded_stl(solid):
    text = mesh64(solid).export(file_type="stl_ascii")
    return trimesh.load_mesh(io.BytesIO(text.encode()), file_type="stl", process=True)


def union(solids):
    return md.Manifold.batch_boolean(solids, md.OpType.Add)


def slab(bounds, bottom, top):
    lo, hi = bounds
    box = trimesh.creation.box(
        extents=[hi[0] - lo[0] + 2, hi[1] - lo[1] + 2, top - bottom]
    )
    box.apply_translation(
        [(hi[0] + lo[0]) / 2, (hi[1] + lo[1]) / 2, (bottom + top) / 2]
    )
    return native(box)


def structural_parts(complete, boards, config):
    """Green plinth, ivory framework, coral birth stems, violet birth heads.

    Every colored birth corridor follows an emitted transition's actual selected
    source and destination. Corridors and terminal clearance assign birth regions
    and ivory-colored joint rims without changing the complete geometry (standard
    links remain full constant-width sections). Kissing material interfaces
    belong to a full junction neighborhood, never a physical gap in the shape.
    """
    recipe = config["structural_color"]
    scale = config["cell_mm"]
    zfactor = np.tan(np.radians(config["angle_degrees"])) * np.sqrt(2)
    step = scale * zfactor
    raw = boards_to_mesh(
        boards,
        config["angle_degrees"],
        scale,
        transition=config.get("transition", "thin"),
        transition_overlap_cells=config.get("transition_overlap_cells", 0.0),
    )
    shift = np.array(
        [
            -raw.bounds[:, 0].mean(),
            -raw.bounds[:, 1].mean(),
            config["base_height_mm"] - raw.bounds[0, 2],
        ]
    )
    solid = native(complete)
    births, heads = [], []
    link_counts = {"survival": 0, "birth": 0}
    for z, (old, new) in enumerate(itertools.pairwise(boards)):
        masks = []
        for x, y, vertices in transition_vertices(old, new, z):
            if old[x, y]:
                link_counts["survival"] += 1
                continue
            link_counts["birth"] += 1
            vertices = np.asarray(vertices, dtype=float)
            lower, upper = vertices[:4].copy(), vertices[4:].copy()
            start, stop = recipe["link_interval"]
            vertices = np.vstack(
                [lower * (1 - start) + upper * start, lower * (1 - stop) + upper * stop]
            )
            for cap in [vertices[:4], vertices[4:]]:
                center = cap.mean(axis=0)
                cap[:] = (cap - center) * recipe["corridor_width_cells"] + center
            vertices[:, 2] *= zfactor
            vertices *= scale
            vertices += shift
            masks.append(native(trimesh.convex.convex_hull(vertices)))
        if not masks:
            continue
        generation = union(masks)
        births.append(generation)
        bottom = config["base_height_mm"] + (z + recipe["head_start"]) * step
        top = config["base_height_mm"] + (z + 1) * step
        # Only the final born volume needs this cut: intersecting the same
        # corridor twice creates coincident side faces in constant-width links.
        heads.append(slab(complete.bounds, bottom, top))
    if not births:
        raise ValueError("Structural recipe needs at least one birth link")
    birth_mask, head_mask = union(births), union(heads)
    base_mask = slab(
        complete.bounds,
        complete.bounds[0, 2] - 1,
        config["base_height_mm"] + recipe["plinth_overlap_mm"],
    )
    green, above = solid ^ base_mask, solid - base_mask
    born = above ^ birth_mask
    ivory = above - birth_mask
    coral, violet = born - head_mask, born ^ head_mask
    parts = [green, ivory, coral, violet]
    junction_edges = []
    junction_material = 1
    # STL has no vertex identities: kissing same-color sheets must not weld into
    # nonmanifold edges on reload. Bound the repair to two material-only passes.
    for attempt in range(2):
        boxes = []
        decoded = [decoded_stl(part) for part in parts]
        if attempt == 0 and not decoded[1].is_watertight:
            junction_material = 0
        for loaded in decoded:
            counts = np.bincount(loaded.edges_unique_inverse)
            for edge in loaded.edges_unique[counts > 2]:
                points = loaded.vertices[edge]
                junction_edges.append(points.tolist())
                clearance = recipe["junction_clearance_mm"]
                direction = points[1] - points[0]
                length = np.linalg.norm(direction)
                if length < 1e-9:
                    direction = np.array([0.271, 0.439, 0.857])
                    zaxis = direction / np.linalg.norm(direction)
                else:
                    zaxis = direction / length
                xaxis = np.cross(zaxis, [0.371, 0.529, 0.763])
                xaxis /= np.linalg.norm(xaxis)
                yaxis = np.cross(zaxis, xaxis)
                box = trimesh.creation.box(
                    extents=[2 * clearance, 2 * clearance, length + 2 * clearance]
                )
                transform = np.eye(4)
                transform[:3, :3] = np.stack([xaxis, yaxis, zaxis], axis=1)
                transform[:3, 3] = points.mean(axis=0)
                box.apply_transform(transform)
                boxes.append(native(box))
        if not boxes:
            break
        fence = union(boxes)
        # Full complete-shape neighborhoods belong to an unbroken material; there
        # is no physical gap, added support, surface paint, or new shape here.
        parts = [
            part - fence if i != junction_material else part + (solid ^ fence)
            for i, part in enumerate(parts)
        ]
    loaded = [decoded_stl(part) for part in parts]
    if not all(m.is_volume and m.is_watertight for m in loaded):
        raise ValueError(
            "Color interfaces did not reload as closed positive-volume bodies"
        )
    return loaded, {
        "link_counts": link_counts,
        "junction_edges": junction_edges,
        "junction_pass_limit": 2,
        "junction_material_index": junction_material,
        "stl_encoding": "ASCII float64",
        "construction_source_is_not_unique_ancestry": True,
    }


def partition_certificate(complete, parts):
    """Coverage proof without a fragile N-way weld of coincident interfaces.

    Each part is inside the complete solid; pairwise intersections have zero
    volume; summed volume equals the complete volume. These imply union equality.
    Pairwise overlap sums bound multiply-counted volume even for triple overlaps.
    """
    authority = native(complete)
    solids = [native(part) for part in parts]
    outside = [abs((part - authority).volume()) for part in solids]
    overlaps = [abs((a ^ b).volume()) for a, b in itertools.combinations(solids, 2)]
    delta = sum(part.volume for part in parts) - complete.volume
    added_bound = sum(outside)
    missing_bound = max(0, -delta) + added_bound + sum(overlaps)
    tolerance = max(0.01, complete.volume * 1e-6)
    assert max(added_bound, missing_bound, abs(delta), sum(overlaps)) < tolerance, (
        added_bound,
        missing_bound,
        delta,
        overlaps,
    )
    bounds = np.array(
        [
            np.vstack([p.bounds[0] for p in parts]).min(axis=0),
            np.vstack([p.bounds[1] for p in parts]).max(axis=0),
        ]
    )
    assert np.allclose(bounds, complete.bounds, atol=2e-5)
    return {
        "method": "subset + pairwise nonoverlap + volume conservation",
        "outside_by_material_mm3": outside,
        "pairwise_overlaps_mm3": overlaps,
        "summed_volume_delta_mm3": float(delta),
        "union_added_bound_mm3": float(added_bound),
        "union_missing_bound_mm3": float(missing_bound),
        "equivalence_tolerance_mm3": float(tolerance),
    }


def section_certificate(complete, parts, config):
    """Measure interleaved material areas within identical horizontal levels."""
    authority = native(complete)
    solids = [native(p) for p in parts]
    step = config["cell_mm"] * np.tan(np.radians(config["angle_degrees"])) * np.sqrt(2)
    rows = []
    for generation in range(config["generations"]):
        for phase in [0.5, 0.85]:
            height = config["base_height_mm"] + (generation + phase) * step
            area = authority.slice(height).area()
            areas = [p.slice(height).area() for p in solids]
            assert abs(sum(areas) - area) < max(0.001, area * 1e-5)
            rows.append(
                {
                    "time": generation + phase,
                    "z_mm": height,
                    "areas_mm2": areas,
                    "materials_over_one_percent": sum(a > area * 0.01 for a in areas),
                }
            )
    assert (
        sum(r["materials_over_one_percent"] >= 2 for r in rows) >= config["generations"]
    )
    return rows
