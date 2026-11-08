"""Constant-width links use the original standard constructor, not thin links."""

import json
from pathlib import Path

import numpy as np
import pytest
import trimesh

import conway3d
from conway3d import boards_to_mesh, transition_vertices
from conway_to_stl import conway_to_boards
from sculptures import build_design, integrated_base
from trimesh_conway import get_parallelipiped_trimesh, native, union_meshes

SEEDS = Path(__file__).resolve().parents[1] / "seeds/sculptures"
STANDARD = json.loads((SEEDS / "intricate-standard.json").read_text())
THIN = json.loads((SEEDS / "intricate.json").read_text())


@pytest.mark.parametrize("delta", [(0, 0), (1, -1)])
@pytest.mark.parametrize("overlap", [0, 0.0001])
def test_standard_sections_are_constant_without_inflated_caps(delta, overlap):
    old = np.zeros((5, 5), dtype=bool)
    old[2, 2] = True
    new = np.zeros_like(old)
    new[2 + delta[0], 2 + delta[1]] = True
    mesh = boards_to_mesh(
        [old, new],
        angle=22,
        scale=6,
        transition="standard",
        transition_overlap_cells=overlap,
    )
    solid = native(mesh)
    height = 6 * np.tan(np.radians(22)) * np.sqrt(2)
    width = 6 * (1 + overlap)
    for phase in [1e-6, 0.05, 0.25, 0.5, 0.75, 0.95, 1 - 1e-6]:
        section = solid.slice(height * phase)
        points = np.concatenate(section.to_polygons())
        assert np.allclose(np.ptp(points, axis=0), [width, width], atol=1e-10)
        assert np.isclose(section.area(), width**2, atol=1e-10)
    assert np.allclose(mesh.extents[:2], width + 6 * np.abs(delta))
    assert np.isclose(mesh.volume, width**2 * height)


def test_pipeline_calls_original_eight_vertex_constructor(monkeypatch):
    old = np.zeros((5, 5), dtype=bool)
    old[2, 2] = True
    new = old.copy()
    new[1, 3] = True
    actual = []

    def capture(vertices):
        actual.append(vertices.copy())
        result = get_parallelipiped_trimesh(vertices)
        assert len(result.thing.vertices) == 8 and len(result.thing.faces) == 12
        assert np.array_equal(result.thing.vertices, vertices)
        return result

    def forbidden(*args, **kwargs):
        raise AssertionError("Standard geometry must not call the thin constructor")

    monkeypatch.setattr(conway3d, "get_parallelipiped_trimesh", capture)
    monkeypatch.setattr(conway3d, "get_thin_transition_trimesh", forbidden)
    boards_to_mesh([old, new], transition="standard")
    expected = [np.asarray(v) for _, _, v in transition_vertices(old, new, 0)]
    assert len(actual) == len(expected) == 2
    assert all(np.array_equal(a, b) for a, b in zip(actual, expected))


@pytest.mark.parametrize(
    "recipe,prior", list(zip(STANDARD["designs"], THIN["designs"]))
)
def test_standard_recipe_changes_only_transition_identity(recipe, prior):
    exempt = {
        "id",
        "title",
        "subtitle",
        "transition",
        "transition_overlap_cells",
        "structural_color",
    }
    assert {k: v for k, v in recipe.items() if k not in exempt} == {
        k: v for k, v in prior.items() if k not in exempt
    }
    assert recipe["id"] == prior["id"] + "-standard"
    assert {
        k: v
        for k, v in recipe["structural_color"].items()
        if k != "corridor_width_cells"
    } == {
        k: v
        for k, v in prior["structural_color"].items()
        if k != "corridor_width_cells"
    }
    assert recipe["structural_color"]["corridor_width_cells"] == 1.001
    assert recipe["transition"] == "standard"
    assert recipe["transition_overlap_cells"] == 0.0001
    # Same exact boards, not only matching population totals.
    bs = conway_to_boards(
        SEEDS / recipe["seed"],
        recipe["generations"],
        world_padding=recipe["world_padding"],
    )[0]
    prior_bs = conway_to_boards(
        SEEDS / prior["seed"],
        prior["generations"],
        world_padding=prior["world_padding"],
    )[0]
    assert all(np.array_equal(a, b) for a, b in zip(bs, prior_bs))


def test_standard_galaxy_overlap_closes_edge_contacts_on_stl_reload(tmp_path):
    config = STANDARD["designs"][0]
    bs = conway_to_boards(
        SEEDS / config["seed"],
        config["generations"],
        world_padding=config["world_padding"],
    )[0]
    mesh = boards_to_mesh(
        bs,
        config["angle_degrees"],
        config["cell_mm"],
        config["transition"],
        config["transition_overlap_cells"],
    )
    mesh.apply_translation([-mesh.bounds[:, 0].mean(), -mesh.bounds[:, 1].mean(), 4])
    mesh = union_meshes([mesh, integrated_base(mesh, 4, 5)])
    path = tmp_path / "standard.stl"
    mesh.export(path, file_type="stl_ascii")
    loaded = trimesh.load_mesh(path)
    assert loaded.is_volume and loaded.is_watertight and loaded.is_winding_consistent
    assert len(loaded.split(only_watertight=False, repair=False)) == 1
    assert np.allclose(loaded.extents, [88.0006, 88.0006, 113.7048386810548])


def test_standard_metadata_and_coloring_do_not_change_complete_solid(tmp_path):
    config = {**STANDARD["designs"][0], "generations": 2}
    out, parts = build_design(config, STANDARD["palette"], SEEDS, tmp_path)
    settings = json.loads((out / "settings.json").read_text())
    assert settings["transition"] == "standard"
    assert "thin" not in settings["geometry"]
    assert "constant-width parallelepiped" in settings["geometry"]
    assert settings["transition_overlap_cells"] == 0.0001
    from structural_color import partition_certificate

    partition_certificate(trimesh.load_mesh(out / "complete.stl"), parts)


@pytest.mark.parametrize(
    "kind,overlap",
    [("unknown", 0), ("thin", 0.001), ("standard", -1), ("standard", float("nan"))],
)
def test_invalid_transition_options(kind, overlap):
    old = np.ones((2, 2), dtype=bool)
    with pytest.raises(ValueError):
        boards_to_mesh([old, old], transition=kind, transition_overlap_cells=overlap)


def test_birth_materials_reach_the_standard_link_surface_without_changing_width():
    from structural_color import partition_certificate, structural_parts

    old = np.zeros((5, 5), dtype=bool)
    old[2, 2] = True
    new = np.zeros_like(old)
    new[1, 3] = True
    config = {**STANDARD["designs"][0], "generations": 1}
    mesh = boards_to_mesh(
        [old, new],
        angle=22,
        scale=6,
        transition="standard",
        transition_overlap_cells=0.0001,
    )
    mesh.apply_translation([-mesh.bounds[:, 0].mean(), -mesh.bounds[:, 1].mean(), 4])
    mesh = union_meshes([mesh, integrated_base(mesh, 4, 5)])
    parts, _ = structural_parts(mesh, [old, new], config)
    partition_certificate(mesh, parts)
    step = 6 * np.tan(np.radians(22)) * np.sqrt(2)
    for phase, color in [(0.5, 2), (0.85, 3)]:
        height = 4 + phase * step
        complete_section = native(mesh).slice(height)
        colored_section = native(parts[color]).slice(height)
        assert np.isclose(colored_section.area(), complete_section.area())
        assert np.isclose(colored_section.area(), 6.0006**2)
