"""Construction-root propagation and whole-volume ownership, not height bands."""

import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest
import trimesh

from conway3d import boards_to_mesh, transition_vertices
from conway_to_stl import conway_to_boards
from sculptures import integrated_base
from structural_color import (
    partition_certificate,
    section_certificate,
    strand_families,
    structural_parts,
)
from trimesh_conway import native, union_meshes

SEEDS = Path(__file__).resolve().parents[1] / "seeds/sculptures"
RECIPES = json.loads((SEEDS / "strands.json").read_text())["designs"]


def history(d):
    return conway_to_boards(
        SEEDS / d["seed"], d["generations"], world_padding=d["world_padding"]
    )[0]


@pytest.mark.parametrize(
    "d,root_count,counts",
    [
        (RECIPES[0], 22, [217, 111, 75, 28]),
        (RECIPES[1], 26, [440, 357, 224, 58]),
        (RECIPES[2], 9, [449, 442, 294, 287]),
    ],
)
def test_every_link_inherits_actual_selected_source_without_color_resets(
    d, root_count, counts
):
    boards = history(d)
    records, forest = strand_families(boards)
    assert len(forest["trees"]) == root_count
    assert forest["anchor_link_counts"] == counts
    assert forest["anchor_alive_at_cut"] == [True] * 4
    labels = {tuple(map(int, p)): tuple(map(int, p)) for p in np.argwhere(boards[0])}
    cursor = 0
    families = defaultdict(set)
    times = defaultdict(set)
    for z, (old, new) in enumerate(itertools.pairwise(boards)):
        following = {}
        for x, y, vertices in transition_vertices(old, new, z):
            r = records[cursor]
            cursor += 1
            source = tuple(vertices[0][:2])
            assert r["source"] == list(source) and r["destination"] == [x, y]
            assert tuple(r["root"]) == labels[source]
            following[x, y] = labels[source]
            families[tuple(r["root"])].add(r["family"])
            times[r["family"]].add(z)
        labels = following
    assert cursor == len(records)
    assert all(len(colors) == 1 for colors in families.values())
    assert all(ts == set(range(d["generations"])) for ts in times.values())


def test_evolving_paths_are_not_static_xy_color_buckets():
    records, _ = strand_families(history(RECIPES[0]))
    sites = defaultdict(set)
    for r in records:
        sites[tuple(r["destination"])].add(r["family"])
    assert any(len(colors) > 1 for colors in sites.values())


@pytest.mark.parametrize("d", RECIPES, ids=lambda d: d["id"])
def test_shorter_history_is_exact_prefix_of_previous_standard_history(d):
    previous = json.loads((SEEDS / "intricate-standard.json").read_text())["designs"]
    prior = next(p for p in previous if p["seed"] == d["seed"])
    boards = history(d)
    full = history(prior)
    assert np.array_equal(boards, full[: len(boards)])
    assert all(
        not (b[:2].any() or b[-2:].any() or b[:, :2].any() or b[:, -2:].any())
        for b in boards
    )
    assert d["transition"] == "standard" and d["transition_overlap_cells"] == 0.0001


def test_whole_birth_link_has_one_material_through_every_interior_phase():
    old = np.zeros((5, 5), dtype=bool)
    old[2, 2] = True
    new = np.zeros_like(old)
    new[1, 3] = True
    d = {**RECIPES[0], "generations": 1}
    mesh = boards_to_mesh([old, new], 40, 6, "standard", 0.0001)
    mesh.apply_translation([-mesh.bounds[:, 0].mean(), -mesh.bounds[:, 1].mean(), 2])
    base = integrated_base(mesh, 2, 1.8)
    complete = union_meshes([mesh, base])
    parts, structure = structural_parts(complete, [old, new], d, base)
    assert len(parts) == 1 and structure["forest"]["links"][0]["family"] == 0
    partition_certificate(complete, parts)
    step = 6 * np.tan(np.radians(40)) * np.sqrt(2)
    for phase in [0.05, 0.25, 0.5, 0.64, 0.85, 0.95]:
        assert np.isclose(native(parts[0]).slice(2 + phase * step).area(), 6.0006**2)
    sections = section_certificate(complete, parts, d)
    assert all(r["materials_over_one_percent"] == 1 for r in sections)  # no old quota


def test_real_galaxy_has_closed_long_connected_families_and_conserved_volume(tmp_path):
    d = RECIPES[0]
    boards = history(d)
    mesh = boards_to_mesh(boards, d["angle_degrees"], d["cell_mm"], "standard", 0.0001)
    mesh.apply_translation([-mesh.bounds[:, 0].mean(), -mesh.bounds[:, 1].mean(), 2])
    base = integrated_base(mesh, 2, 1.8)
    complete = union_meshes([mesh, base])
    parts, structure = structural_parts(complete, boards, d, base)
    partition_certificate(complete, parts)
    assert len(parts) == 4
    step = 6 * np.tan(np.radians(40)) * np.sqrt(2)
    for i, part in enumerate(parts):
        path = tmp_path / f"family-{i}.stl"
        part.export(path, file_type="stl_ascii")
        loaded = trimesh.load_mesh(path)
        assert (
            loaded.is_volume and loaded.is_watertight and loaded.is_winding_consistent
        )
        spans = [
            c.extents[2] for c in loaded.split(only_watertight=False, repair=False)
        ]
        assert max(spans) / step > 15.9
    assert all(
        r["radius_mm"] <= 0.00015 + 1e-12 for r in structure["interface_repairs"]
    )


@pytest.mark.parametrize("count", [0, 5])
def test_invalid_family_count(count):
    with pytest.raises(ValueError):
        strand_families(history(RECIPES[0]), count)
