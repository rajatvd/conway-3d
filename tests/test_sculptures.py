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
    validate_assets,
)
from structural_color import partition_certificate, structural_parts
from trimesh_conway import union_meshes

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


@pytest.mark.parametrize(
    "seed,generations,scale",
    [("glider", 4, 4), ("pulsar", 3, 5), ("r-pentomino", 16, 3.8)],
)
def test_full_roundtrip_and_color_partition(tmp_path, seed, generations, scale):
    mesh = boards_to_mesh(boards(seed, generations), scale=scale)
    mesh.apply_translation([-mesh.bounds[:, 0].mean(), -mesh.bounds[:, 1].mean(), 3.2])
    mesh = union_meshes([mesh, integrated_base(mesh, 3.2, 3.5)])
    mesh.export(tmp_path / "complete.stl", file_type="stl_ascii")
    mesh = trimesh.load_mesh(tmp_path / "complete.stl")
    config = next(
        d
        for d in json.loads((SEEDS / "designs.json").read_text())["designs"]
        if d["seed"] == seed + ".txt"
    )
    config = {**config, "generations": generations, "cell_mm": scale}
    config.pop("complete_sha256")
    parts, _ = structural_parts(mesh, boards(seed, generations), config)
    for i, (part, color) in enumerate(zip(parts, PALETTE)):
        part.export(
            tmp_path / f"{i + 1:02}-{color['name'].lower()}.stl", file_type="stl_ascii"
        )
    parts = [
        trimesh.load_mesh(tmp_path / f"{i + 1:02}-{c['name'].lower()}.stl")
        for i, c in enumerate(PALETTE)
    ]
    export_3mf(parts, PALETTE, tmp_path / "multicolor.3mf")
    export_glb(parts, PALETTE, tmp_path / "multicolor.glb")
    (tmp_path / "settings.json").write_text(json.dumps({**config, "palette": PALETTE}))
    proof = validate_assets(tmp_path)
    assert proof["complete"]["components"] == 1
    assert proof["three_mf_registered_materials_and_buffers"]
    assert proof["glb_y_up_meters_colors_and_buffers"]
    assert proof["lib3mf_strict"]["mesh_objects"] == 4
    assert all(
        r["materials_over_one_percent"] >= 2 for r in proof["horizontal_sections"]
    )


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


@pytest.mark.parametrize("count", [1, 2, 3, 4])
def test_material_count_roundtrips_without_id_collision(tmp_path, count):
    # Axis slabs are a minimal serialization fixture, not an available art mode.
    parts = []
    for i in range(count):
        m = trimesh.creation.box(extents=[1, 2, 2])
        m.apply_translation([i + 0.5, 0, 1])
        parts.append(m)
    complete = trimesh.creation.box(extents=[count, 2, 2])
    complete.apply_translation([count / 2, 0, 1])
    complete.export(tmp_path / "complete.stl")
    palette = PALETTE[:count]
    for i, (part, color) in enumerate(zip(parts, palette)):
        part.export(tmp_path / f"{i + 1:02}-{color['name'].lower()}.stl")
    parts = [
        trimesh.load_mesh(tmp_path / f"{i + 1:02}-{c['name'].lower()}.stl")
        for i, c in enumerate(palette)
    ]
    export_3mf(parts, palette, tmp_path / "multicolor.3mf")
    export_glb(parts, palette, tmp_path / "multicolor.glb")
    (tmp_path / "settings.json").write_text(json.dumps({"palette": palette}))
    proof = validate_assets(tmp_path)
    assert proof["lib3mf_strict"]["mesh_objects"] == count
    assert proof["lib3mf_strict"]["warnings"] == 0


@pytest.mark.parametrize("error", ["hole", "overlap", "outside"])
def test_partition_certificate_rejects_bad_geometry(error):
    full = trimesh.creation.box(extents=[4, 4, 4])
    if error == "hole":
        parts = [trimesh.creation.box(extents=[2, 2, 2])]
    elif error == "overlap":
        parts = [full.copy(), full.copy()]
    else:
        shifted = full.copy()
        shifted.apply_translation([1, 0, 0])
        parts = [shifted]
    with pytest.raises(AssertionError):
        partition_certificate(full, parts)


@pytest.mark.parametrize("exporter", [export_3mf, export_glb])
def test_invalid_material_counts(exporter, tmp_path):
    with pytest.raises(ValueError):
        exporter([], [], tmp_path / "empty")
    with pytest.raises(ValueError):
        exporter([trimesh.creation.box()], PALETTE, tmp_path / "mismatch")


def test_recipe_historical_hash_is_enforced_for_copy_not_reserialization(tmp_path):
    import hashlib

    from sculptures import build_design

    config = json.loads((SEEDS / "designs.json").read_text())["designs"][0]
    config = {**config, "generations": 4, "complete_sha256": "0" * 64}
    output = tmp_path / "generated"
    directory, _ = build_design(config, PALETTE, SEEDS, output)
    actual = hashlib.sha256((directory / "complete.stl").read_bytes()).hexdigest()
    assert actual != config["complete_sha256"]
    assert (
        json.loads((directory / "settings.json").read_text())["complete_sha256"]
        == actual
    )
    with pytest.raises(ValueError, match="authoritative STL"):
        build_design(config, PALETTE, SEEDS, tmp_path / "copied", complete_from=output)
