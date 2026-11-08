"""Small independent 2D proofs for the intricate recipes; no mesh sweep."""

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from scipy.ndimage import label

from conway3d import conway_step
from conway_to_stl import conway_to_boards

SEEDS = Path(__file__).resolve().parents[1] / "seeds/sculptures"
RECIPES = json.loads((SEEDS / "intricate.json").read_text())["designs"]


def history(recipe):
    return conway_to_boards(
        SEEDS / recipe["seed"],
        recipe["generations"],
        world_padding=recipe["world_padding"],
    )[0]


@pytest.mark.parametrize("recipe", RECIPES, ids=lambda d: d["id"])
def test_every_selected_generation_against_sparse_unbounded_life(recipe):
    boards = history(recipe)
    live = set(map(tuple, np.argwhere(boards[0])))
    for board in boards:
        assert set(map(tuple, np.argwhere(board))) == live
        assert not (
            board[:2].any()
            or board[-2:].any()
            or board[:, :2].any()
            or board[:, -2:].any()
        )
        neighbors = Counter(
            (x + dx, y + dy)
            for x, y in live
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            if dx or dy
        )
        live = {p for p, n in neighbors.items() if n == 3 or (n == 2 and p in live)}
    assert recipe["cell_mm"] >= 3.8  # complexity is not obtained by microscopic cells


def test_galaxy_has_four_complete_period_eight_cycles():
    boards = history(RECIPES[0])
    assert [int(b.sum()) for b in boards[:9]] == [28, 32, 48, 40, 64, 36, 36, 44, 28]
    assert not any(np.array_equal(boards[0], boards[t]) for t in range(1, 8))
    assert all(np.array_equal(boards[t], boards[t + 8]) for t in range(25))
    assert np.array_equal(boards[0], boards[32])
    assert all(np.array_equal(board, np.rot90(board)) for board in boards)


def test_gun_repeats_its_core_and_emits_a_real_glider():
    recipe = RECIPES[1]
    p = recipe["world_padding"]
    boards = conway_to_boards(SEEDS / recipe["seed"], 90, world_padding=p)[0]
    core = [b[p : p + 9, p : p + 36] for b in boards]
    assert all(np.array_equal(core[t], core[t + 30]) for t in range(61))
    assert [int(boards[t].sum()) for t in [0, 30, 60, 90]] == [36, 41, 46, 51]
    outside = boards[60].copy()
    outside[p : p + 9, p : p + 36] = False
    components, count = label(outside, structure=np.ones((3, 3)))
    gliders = 0
    for i in range(1, count + 1):
        isolated = components == i
        if isolated.sum() != 5:
            continue
        evolved = isolated.copy()
        for _ in range(4):
            evolved = conway_step(evolved)
        gliders += np.array_equal(evolved, np.roll(np.roll(isolated, 1, 0), 1, 1))
    assert gliders >= 1


def test_authored_crossfire_is_an_interaction_not_cosmetic_duplication():
    recipe = RECIPES[2]
    board = history(recipe)[0]
    acorn = np.array(
        [[0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0], [1, 1, 0, 0, 1, 1, 1]],
        dtype=bool,
    )
    p = recipe["world_padding"]
    a, b = np.zeros_like(board), np.zeros_like(board)
    a[p : p + 3, p : p + 7] = acorn
    b[p : p + 3, p + 12 : p + 19] = np.rot90(acorn, 2)
    assert np.array_equal(a | b, board)
    differences = []
    for t in range(33):
        if not np.array_equal(board, a | b):
            differences.append(t)
        board, a, b = conway_step(board), conway_step(a), conway_step(b)
    assert differences[0] == 11
    populations = [int(b.sum()) for b in history(recipe)]
    assert max(populations) == 110 and populations.index(110) == 27
    assert populations[-1] == 78


def test_large_gun_complete_mesh_retains_double_precision_through_stl(tmp_path):
    import trimesh

    from conway3d import boards_to_mesh
    from sculptures import integrated_base
    from trimesh_conway import union_meshes

    recipe = RECIPES[1]
    mesh = boards_to_mesh(history(recipe), recipe["angle_degrees"], recipe["cell_mm"])
    mesh.apply_translation(
        [-mesh.bounds[:, 0].mean(), -mesh.bounds[:, 1].mean(), recipe["base_height_mm"]]
    )
    mesh = union_meshes(
        [
            mesh,
            integrated_base(mesh, recipe["base_height_mm"], recipe["base_padding_mm"]),
        ]
    )
    path = tmp_path / "complete.stl"
    mesh.export(path, file_type="stl_ascii")
    loaded = trimesh.load_mesh(path, process=True)
    assert np.isfinite(loaded.vertices).all()
    assert loaded.is_watertight and loaded.is_winding_consistent and loaded.is_volume
    assert len(loaded.split(only_watertight=False, repair=False)) == 1
    assert np.allclose(loaded.extents, [94.6, 154.6, 121.63349532024893])
