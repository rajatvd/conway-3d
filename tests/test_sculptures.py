import itertools
import json
from pathlib import Path

import numpy as np
import pytest
import trimesh
from PIL import Image

import conway3d
from conway3d import boards_to_mesh, conway_step, transition_vertices
from conway_to_stl import conway_to_boards
from sculpture_preview import checked_jpeg
from sculptures import (
    export_3mf,
    export_glb,
    integrated_base,
    split_epochs,
    validate_assets,
)

SEEDS = Path(__file__).resolve().parents[1] / "seeds/sculptures"
PALETTE = json.loads((SEEDS / "designs.json").read_text())["palette"]


def boards(name, generations):
    return conway_to_boards(SEEDS / f"{name}.txt", generations, world_padding=20)[0]


def test_glider_translation():
    bs = boards("glider", 4)
    assert np.array_equal(bs[-1], np.roll(np.roll(bs[0], 1, 0), 1, 1))
    assert [b.sum() for b in bs] == [5] * 5


def test_pulsar_period():
    bs = boards("pulsar", 3)
    assert [b.sum() for b in bs] == [48, 56, 72, 48]
    assert np.array_equal(bs[0], bs[3])


def test_r_pentomino_transient():
    bs = boards("r-pentomino", 4)
    assert [b.sum() for b in bs] == [5, 6, 7, 9, 8]


@pytest.mark.parametrize(
    "seed,generations", [("glider", 16), ("pulsar", 9), ("r-pentomino", 16)]
)
def test_link_sources_match_original_traversal(seed, generations):
    bs = boards(seed, generations)
    for z, (old, new) in enumerate(itertools.pairwise(bs)):
        emitted = list(transition_vertices(old, new, z))
        assert len(emitted) == new.sum()
        for x, y, vertices in emitted:
            expected = (x, y)
            if not old[x, y]:
                for i in (-1, 0, 1):
                    for j in (-1, 0, 1):
                        if (
                            (i or j)
                            and 0 <= x + i < old.shape[0]
                            and 0 <= y + j < old.shape[1]
                            and old[x + i, y + j]
                        ):
                            expected = (x + i, y + j)
                            break
            assert vertices[0] == [*expected, z]
            assert vertices[4] == [x, y, z + 1]


def test_rectangular_links():
    old = np.zeros((5, 9), bool)
    old[2, 6:9] = True
    links = list(transition_vertices(old, conway_step(old), 0))
    assert {(x, y) for x, y, _ in links} == {(1, 7), (2, 7), (3, 7)}


def test_disconnected_birth_rejected():
    old = np.zeros((4, 4), bool)
    new = old.copy()
    new[0, 0] = True
    with pytest.raises(ValueError, match="no local source"):
        list(transition_vertices(old, new, 0))


@pytest.mark.parametrize("angle,scale", [(0, 1), (90, 1), (45, 0)])
def test_invalid_dimensions(angle, scale):
    with pytest.raises(ValueError):
        boards_to_mesh(boards("glider", 1), angle=angle, scale=scale)


def test_empty_export():
    with pytest.raises(ValueError, match="No cells"):
        boards_to_mesh([np.zeros((3, 3), bool)] * 2)


def test_scratch_does_not_overwrite_cwd(tmp_path, monkeypatch):
    # Exercise the real CSG route, not a mocked filename.
    monkeypatch.chdir(tmp_path)
    sentinel = tmp_path / "m.stl"
    sentinel.write_bytes(b"preserve this model")
    live = np.zeros((3, 3), bool)
    live[1, 1] = True
    paths = []
    original = conway3d.get_transition

    def capture(vertices):
        thing = original(vertices)
        export = thing.to_stl

        def record(path):
            paths.append(Path(path))
            return export(path)

        thing.to_stl = record
        return thing

    monkeypatch.setattr(conway3d, "get_transition", capture)
    mesh = conway3d.boards_to_mesh_with_lego([live, live], scale=2)
    assert mesh.is_volume
    assert sentinel.read_bytes() == b"preserve this model"
    assert len(paths) == 1 and paths[0].parent != tmp_path
    assert not paths[0].exists()  # scratch lifetime ends after load


def test_full_roundtrip_and_color_partition(tmp_path):
    mesh = boards_to_mesh(boards("glider", 4), scale=4)
    mesh.apply_translation([0, 0, 3.2])
    mesh = trimesh.boolean.union(
        [mesh, integrated_base(mesh, 3.2, 3.5)], engine="manifold"
    )
    mesh.export(tmp_path / "complete.stl")
    mesh = trimesh.load_mesh(tmp_path / "complete.stl")
    parts = split_epochs(mesh, [3.2 + 1.5 * 4 * np.sqrt(2), 3.2 + 2.5 * 4 * np.sqrt(2)])
    for i, (part, color) in enumerate(zip(parts, PALETTE)):
        part.export(tmp_path / f"{i + 1:02}-{color['name'].lower()}.stl")
    parts = [
        trimesh.load_mesh(tmp_path / f"{i + 1:02}-{c['name'].lower()}.stl")
        for i, c in enumerate(PALETTE)
    ]
    export_3mf(parts, PALETTE, tmp_path / "multicolor.3mf")
    export_glb(parts, PALETTE, tmp_path / "multicolor.glb")
    (tmp_path / "settings.json").write_text(json.dumps({"palette": PALETTE}))
    proof = validate_assets(tmp_path)
    assert proof["complete"]["components"] == 1
    assert proof["three_mf_registered_materials_and_buffers"]
    assert proof["glb_y_up_meters_colors_and_buffers"]


def test_preview_guards(tmp_path):
    p = tmp_path / "valid.jpg"
    Image.new("RGB", (640, 200)).save(p)
    assert checked_jpeg(p)["pixels"] == [640, 200]
    p = tmp_path / "too-big.jpg"
    Image.new("RGB", (641, 200)).save(p)
    with pytest.raises(AssertionError):
        checked_jpeg(p)
    p = tmp_path / "not-jpeg.jpg"
    Image.new("RGB", (100, 100)).save(p, format="PNG")
    with pytest.raises(AssertionError):
        checked_jpeg(p)
