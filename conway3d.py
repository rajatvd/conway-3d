import numpy as np
from functools import partial
import matplotlib.pyplot as plt
import trimesh
from pathlib import Path
from tempfile import TemporaryDirectory

from trimesh_conway import (
    get_parallelipiped_trimesh,
    get_thin_transition_trimesh,
    make_lego_trimesh,
    union_meshes,
)

from csg_conway import get_parallelipiped_csg, get_thin_transition_csg, make_lego_csg


def conway_step(board):
    # Count the number of neighbors for each cell
    neighbors = sum(
        np.roll(np.roll(board, i, 0), j, 1)
        for i in (-1, 0, 1)
        for j in (-1, 0, 1)
        if (i != 0 or j != 0)
    )
    # Apply Conway's rules
    return (neighbors == 3) | (board & (neighbors == 2))


def conway(board, steps):
    for _ in range(steps):
        board = conway_step(board)
    return board


def point_to_index(x, y, z, nx, ny, nz):
    return x * ny * nz + y * nz + z


# %%
make_lego = make_lego_trimesh
get_transition = get_thin_transition_trimesh
# get_transition = get_parallelipiped_trimesh

# %%
make_lego = make_lego_csg
# get_transition = get_parallelipiped_csg
get_transition = get_thin_transition_csg


def transition_vertices(old, new, z):
    """Yield the original cell-to-cell birth/survival links for one time step.

    Births use the legacy traversal's last occupied neighbor row, first
    occupied column. Dead cells emit no link. Coordinates are (row, column, t).
    """
    if old.shape != new.shape or old.ndim != 2:
        raise ValueError("Transition boards must be matching 2D arrays")
    nx, ny = old.shape
    for x, y in np.argwhere(new):
        oldx, oldy = x, y
        if not old[x, y]:
            sources = []
            for i in (-1, 0, 1):
                for j in (-1, 0, 1):
                    if (
                        (i or j)
                        and 0 <= x + i < nx
                        and 0 <= y + j < ny
                        and old[x + i, y + j]
                    ):
                        sources.append((x + i, y + j))
                        break
            if not sources:
                raise ValueError("Birth has no local source; increase world padding")
            oldx, oldy = sources[-1]
        yield (
            x,
            y,
            [
                [oldx, oldy, z],
                [oldx + 1, oldy, z],
                [oldx + 1, oldy + 1, z],
                [oldx, oldy + 1, z],
                [x, y, z + 1],
                [x + 1, y, z + 1],
                [x + 1, y + 1, z + 1],
                [x, y + 1, z + 1],
            ],
        )


def boards_to_mesh(
    boards, angle=45, scale=1, transition="thin", transition_overlap_cells=0.0
):
    """Batch-union thin or standard constant-width links, without Lego connectors.

    Standard links use the original eight-vertex parallelepiped constructor.
    Optional overlap expands both XY caps uniformly, never time or section shape.

    Like the original CSG exporter, time 0 is the source plane and the last
    board is a finite cut. No artificial initial/final population is added.
    """
    if len(boards) < 2 or not 0 < angle < 90 or scale <= 0:
        raise ValueError("Need two boards, 0 < angle < 90, and positive scale")
    if transition not in ("thin", "standard"):
        raise ValueError("Transition must be thin or standard")
    if not np.isfinite(transition_overlap_cells) or transition_overlap_cells < 0:
        raise ValueError("Transition overlap must be finite and nonnegative")
    if transition != "standard" and transition_overlap_cells:
        raise ValueError("Uniform overlap is only supported for standard transitions")
    constructor = (
        get_parallelipiped_trimesh
        if transition == "standard"
        else get_thin_transition_trimesh
    )
    links = []
    for z, (old, new) in enumerate(zip(boards, boards[1:])):
        for _, _, vertices in transition_vertices(old, new, z):
            vertices = np.asarray(vertices, dtype=float)
            if transition_overlap_cells:
                for cap in (vertices[:4], vertices[4:]):
                    center = cap[:, :2].mean(axis=0)
                    cap[:, :2] = (cap[:, :2] - center) * (
                        1 + transition_overlap_cells
                    ) + center
            links.append(constructor(vertices).thing)
    if not links:
        raise ValueError("No cells in the board")
    mesh = union_meshes(links)
    mesh.vertices[:, 2] *= np.tan(np.radians(angle)) * np.sqrt(2)
    mesh.vertices *= scale
    return mesh


# %%
# given a sequence of boards, construct 3d model with z axis as the time axis
def boards_to_mesh_with_lego(
    boards,
    angle=45,
    base_lego_points=[],
    top_lego_points=[],
    lego_width=0.4,
    lego_height=0.4,
    scale=1,
    top_lego_scale=0.9,
):
    # convert to list of tuples
    base_lego_points = [tuple(p) for p in base_lego_points]
    top_lego_points = [tuple(p) for p in top_lego_points]
    zs = np.arange(len(boards), dtype=np.float32)

    m = None

    BASE_LEGO_COUNT = 0
    base_legos = None

    # now add transition faces
    for zind in range(len(boards) - 1):
        z = zs[zind]
        for x, y, new_vertices in transition_vertices(
            boards[zind], boards[zind + 1], z
        ):
            new_parallepiped = get_transition(new_vertices)
            if m is not None:
                print(
                    x,
                    y,
                    z,
                    f"m.is_watertight: {m.is_watertight()}",
                )
                old_watertight = m.is_watertight()
                new_m = m + new_parallepiped
                new_watertight = new_m.is_watertight()
                m = new_m
            else:
                m = new_parallepiped

        if z == 0:
            for x, y in base_lego_points:
                print(f"Adding base lego number {BASE_LEGO_COUNT} at {x}, {y}, {z}")
                bot = zs[0]
                base_lego = make_lego(
                    x,
                    y,
                    bot,
                    lego_height,
                    lego_width,
                    scale=1.1,
                    sign=-1,
                )
                m = m * base_lego
                # if base_legos is not None:
                #     base_legos = base_legos + base_lego
                # else:
                #     base_legos = base_lego
                BASE_LEGO_COUNT += 1

    if m is None:
        raise ValueError("No cells in the board")

    # if base_legos is not None:
    #     m = m * base_legos

    # add lego to the tops
    top = zs[-1]
    for x, y in top_lego_points:
        top_lego = make_lego(
            x,
            y,
            top,
            lego_height,
            lego_width,
            scale=top_lego_scale,
            sign=1,
        )
        if m is not None:
            m = m + top_lego

    # m.vertices[:, 2] *= z_factor
    # m.vertices *= scale
    # CSG exports through a file; never overwrite a caller's model or race
    # another export in the same working directory.
    with TemporaryDirectory(prefix="conway3d-") as scratch:
        path = Path(scratch) / "mesh.stl"
        m.to_stl(str(path))
        t = trimesh.load_mesh(path, process=True)
    print(f"Is watertight: {t.is_watertight}")

    z_factor = np.tan(np.radians(angle)) * np.sqrt(2)

    t.vertices[:, 2] *= z_factor
    t.vertices *= scale

    return t


# %%
def add_base(tm, base_height=2.0, base_padding=5.0, fudge=1e-2):
    # get min and max x and y and add padding
    min_x = np.min(tm.vertices[:, 0])
    min_y = np.min(tm.vertices[:, 1])
    max_x = np.max(tm.vertices[:, 0])
    max_y = np.max(tm.vertices[:, 1])

    min_x = min_x - base_padding
    min_y = min_y - base_padding
    max_x = max_x + base_padding
    max_y = max_y + base_padding

    base_bottom = np.min(tm.vertices[:, 2]) - base_height
    base_top = base_bottom + base_height + fudge

    base_vertices = np.array(
        [
            [min_x, min_y, base_bottom],
            [max_x, min_y, base_bottom],
            [max_x, max_y, base_bottom],
            [min_x, max_y, base_bottom],
            [min_x, min_y, base_top],
            [max_x, min_y, base_top],
            [max_x, max_y, base_top],
            [min_x, max_y, base_top],
        ]
    )

    # the base is a cuboid so it has 6 faces so 12 triangles
    base_faces = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
        ]
    )

    base = trimesh.Trimesh(vertices=base_vertices, faces=base_faces)

    tm = tm.union(base)
    return tm


# %%
def base_span(boards, padding=1):
    N = len(boards[0])
    base = np.zeros((N, N), dtype=bool)
    # base has to be a rectangle
    # find the min and max x and y across all boards and add padding
    min_x = N
    min_y = N
    max_x = 0
    max_y = 0
    for b in boards:
        x, y = np.where(b)
        if len(x) == 0:
            continue
        min_x = min(min_x, min(x))
        min_y = min(min_y, min(y))
        max_x = max(max_x, max(x))
        max_y = max(max_y, max(y))

    min_x = max(0, min_x - padding)
    min_y = max(0, min_y - padding)
    max_x = min(N, max_x + padding)
    max_y = min(N, max_y + padding)
    base[min_x:max_x, min_y:max_y] = True
    return base
