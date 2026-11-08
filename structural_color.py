"""Whole-link material families carried by the actual geometric construction forest.

Selected sources are deterministic construction choices, not biological ancestry.
Crossing ownership never changes the complete solid or resets colors by height.
"""

import io
import itertools
from collections import Counter, defaultdict

import manifold3d as md
import numpy as np
import trimesh

from conway3d import boards_to_mesh, scale_mesh, transition_meshes, transition_vertices
from trimesh_conway import mesh64, native


def decoded_stl(solid):
    text = mesh64(solid).export(file_type="stl_ascii")
    return trimesh.load_mesh(io.BytesIO(text.encode()), file_type="stl", process=True)


def union(solids):
    return md.Manifold.batch_boolean(solids, md.OpType.Add)


def strand_families(boards, count=4):
    """Prefer dominant cut-surviving roots; group minor trees by trail proximity.

    Proximity compares actual cells at shared times, not fixed coordinate bins.
    A tree's family is selected once and inherited through every selected source.
    """
    if not 1 <= count <= 4:
        raise ValueError("Need one to four strand families")
    labels = {tuple(map(int, p)): tuple(map(int, p)) for p in np.argwhere(boards[0])}
    counts, sites, records = Counter(), defaultdict(lambda: defaultdict(set)), []
    for z, (old, new) in enumerate(itertools.pairwise(boards)):
        following = {}
        for x, y, vertices in transition_vertices(old, new, z):
            source = tuple(map(int, vertices[0][:2]))
            destination = (int(x), int(y))
            root = labels[source]
            following[destination] = root
            counts[root] += 1
            sites[root][z].add(source)
            sites[root][z + 1].add(destination)
            records.append((z, source, destination, root))
        labels = following
    if not counts:
        raise ValueError("No nonempty construction roots")
    alive = set(labels.values())
    anchors = sorted(counts, key=lambda r: (r not in alive, -counts[r], r))[:count]

    def distance(a, b):
        return min(
            (x - u) ** 2 + (y - v) ** 2
            for t, points in sites[a].items()
            for x, y in points
            for u, v in sites[b].get(t, ())
        )

    families = {
        root: min(range(len(anchors)), key=lambda i: (distance(root, anchors[i]), i))
        for root in counts
    }
    families.update({root: i for i, root in enumerate(anchors)})
    emitted = [
        {
            "time": z,
            "source": list(source),
            "destination": list(destination),
            "root": list(root),
            "family": families[root],
        }
        for z, source, destination, root in records
    ]
    return emitted, {
        "anchors": [list(r) for r in anchors],
        "anchor_link_counts": [counts[r] for r in anchors],
        "anchor_alive_at_cut": [r in alive for r in anchors],
        "trees": [
            {
                "root": list(r),
                "family": families[r],
                "links": counts[r],
                "last_time": max(sites[r]),
                "alive_at_cut": r in alive,
            }
            for r in sorted(counts)
        ],
        "assignment": "one root label inherited through selected source; nearest concurrent trail for minor roots",
        "links": emitted,
    }


def structural_parts(complete, boards, config, base, color_count=4):
    """Union full emitted links by root family; material zero owns the actual base.

    Earlier anchor rank owns crossing volumes. Tiny kissing-edge repairs only
    transfer existing multiple-family overlap to an incident family; they never
    add geometry, cut height bands, or color isolated heads/corridors.
    """
    records, forest = strand_families(boards, color_count)
    groups = [[] for _ in forest["anchors"]]
    raw = boards_to_mesh(
        boards,
        config["angle_degrees"],
        config["cell_mm"],
        config.get("transition", "thin"),
        config.get("transition_overlap_cells", 0),
    )
    shift = [
        -raw.bounds[:, 0].mean(),
        -raw.bounds[:, 1].mean(),
        config["base_height_mm"] - raw.bounds[0, 2],
    ]
    stream = transition_meshes(
        boards,
        config.get("transition", "thin"),
        config.get("transition_overlap_cells", 0),
    )
    for record, (z, destination, source, mesh) in zip(records, stream, strict=True):
        assert (record["time"], record["destination"], record["source"]) == (
            z,
            list(destination),
            list(source),
        )
        groups[record["family"]].append(native(mesh))
    masks = []
    for group in groups:
        mesh = scale_mesh(
            mesh64(union(group)), config["angle_degrees"], config["cell_mm"]
        )
        mesh.apply_translation(shift)
        masks.append(native(mesh))
    base_solid = native(base)
    claimed, parts = base_solid, []
    for mask in masks:
        parts.append(mask - claimed)
        claimed = claimed + mask
    parts[0] = parts[0] + base_solid
    repairs = []
    radius = (
        config["cell_mm"] * max(config.get("transition_overlap_cells", 0), 1e-4) / 4
    )
    for attempt in range(2):
        changed = False
        for i in range(len(parts)):
            loaded = decoded_stl(parts[i])
            counts = np.bincount(loaded.edges_unique_inverse)
            edges = loaded.edges_unique[counts > 2]
            if not len(edges) or radius == 0:
                continue
            boxes = []
            for edge in edges:
                points = loaded.vertices[edge]
                box = trimesh.creation.box(extents=np.ptp(points, axis=0) + 2 * radius)
                box.apply_translation(points.mean(axis=0))
                boxes.append(native(box))
            overlap = masks[i] ^ union([m for j, m in enumerate(masks) if j != i])
            patch = (union(boxes) ^ overlap) - base_solid
            repairs.append(
                {
                    "pass": attempt,
                    "family": i,
                    "edges": len(edges),
                    "radius_mm": radius,
                    "transferred_volume_mm3": patch.volume(),
                }
            )
            parts = [p + patch if j == i else p - patch for j, p in enumerate(parts)]
            changed = True
        if not changed:
            break
    loaded = [decoded_stl(p) for p in parts]
    if not all(m.is_volume and m.is_watertight for m in loaded):
        raise ValueError("Whole-strand interfaces did not reload as closed solids")
    return loaded, {
        "method": "whole-link root strand families",
        "forest": forest,
        "crossing_ownership": "base then dominant anchor rank; incident-family repair restricted to actual overlapping links",
        "interface_repairs": repairs,
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
    """Record section-area agreement without imposing a material-count quota."""
    authority = native(complete)
    solids = [native(p) for p in parts]
    step = config["cell_mm"] * np.tan(np.radians(config["angle_degrees"])) * np.sqrt(2)
    rows = []
    for generation in range(config["generations"]):
        for phase in [0.25, 0.5, 0.75]:
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
    return rows
