import numpy as np
from functools import partial
import matplotlib.pyplot as plt
import trimesh

from oop3d import WatertightThing, CSGThing, TrimeshThing


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


def get_parallelipiped_trimesh(vertices):
    assert len(vertices) == 8, "Need 8 vertices to make a parallepiped"
    faces = [
        [0, 3, 1],
        [1, 3, 2],
        [0, 4, 7],
        [0, 7, 3],
        [4, 5, 6],
        [4, 6, 7],
        [5, 1, 2],
        [5, 2, 6],
        [2, 3, 6],
        [3, 7, 6],
        [0, 1, 5],
        [0, 5, 4],
    ]
    return TrimeshThing(trimesh.Trimesh(vertices, faces, process=False))
    # return o3d.geometry.TriangleMesh(vertices, faces)


def get_thin_transition_trimesh(vertices, border=0.15, alpha=3.5):
    assert len(vertices) == 8, "Need 8 vertices to make a parallepiped"

    v = np.array(vertices)
    center = np.mean(v, axis=0)

    vtop = v[4:]
    vbot = v[:4]
    upd_vtop = (vtop - vtop.mean(axis=0)) * (1 + border) + vtop.mean(axis=0)
    upd_vbot = (vbot - vbot.mean(axis=0)) * (1 + border) + vbot.mean(axis=0)
    updv = np.concatenate([upd_vbot, upd_vtop], axis=0)

    new_v = v - center
    new_v[:, 2] *= 1 - (alpha) * border
    new_v[:, (0, 1)] *= 1 - alpha * border
    new_v += center

    faces = [
        [0, 3, 1],
        [1, 3, 2],
        [4, 5, 7],
        [5, 6, 7],
        [0, 1, 8],
        [1, 9, 8],
        [1, 2, 10],
        [1, 10, 9],
        [2, 3, 11],
        [2, 11, 10],
        [3, 0, 8],
        [3, 8, 11],
        [4, 12, 13],
        [4, 13, 5],
        [5, 13, 14],
        [5, 14, 6],
        [6, 14, 15],
        [6, 15, 7],
        [7, 15, 12],
        [7, 12, 4],
        [9, 10, 13],
        [9, 13, 12],
        [10, 11, 14],
        [10, 14, 13],
        [11, 8, 15],
        [11, 15, 14],
        [8, 9, 12],
        [8, 12, 15],
    ]

    vertices = updv.tolist() + new_v.tolist()

    return TrimeshThing(trimesh.Trimesh(vertices, faces, process=False))


# %%
def make_lego_trimesh(x, y, z, lego_height, lego_width, scale=1.0):
    border = (1 - lego_width) / 2.0
    lego_mid_height = lego_height - lego_width / 2.0

    new_vertices = [
        [x + border, y + border, z],
        [x + 1 - border, y + border, z],
        [x + 1 - border, y + 1 - border, z],
        [x + border, y + 1 - border, z],
        #
        [x + border, y + border, z + lego_mid_height],
        [x + 1 - border, y + border, z + lego_mid_height],
        [x + 1 - border, y + 1 - border, z + lego_mid_height],
        [x + border, y + 1 - border, z + lego_mid_height],
        #
        [x + 0.5, y + 0.5, z + lego_height],
    ]

    new_faces = [
        [0, 2, 1],
        [0, 3, 2],
        [0, 1, 5],
        [0, 5, 4],
        [1, 2, 6],
        [1, 6, 5],
        [2, 3, 7],
        [2, 7, 6],
        [3, 0, 4],
        [3, 4, 7],
        [4, 5, 8],
        [5, 6, 8],
        [6, 7, 8],
        [7, 4, 8],
    ]

    nv = np.array(new_vertices)
    nv = (nv - nv.mean(axis=0)) * scale + nv.mean(axis=0)
    new_vertices = nv.tolist()
    lego = trimesh.Trimesh(new_vertices, new_faces)
    return TrimeshThing(lego)


make_lego = make_lego_trimesh
get_transition = get_thin_transition_trimesh
# get_transition = get_parallelipiped_trimesh


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
    base_scale=1.0,
):
    # convert to list of tuples
    base_lego_points = [tuple(p) for p in base_lego_points]
    top_lego_points = [tuple(p) for p in top_lego_points]
    # angle is overhang angle in degrees
    # Create grid of vertices corresponding to cell corners
    N = len(boards[0]) + 1
    zs = np.arange(len(boards), dtype=np.float32)
    zs[1:] -= 1 - base_scale

    vertices = np.array([[x, y, z] for x in range(N) for y in range(N) for z in zs])
    vertices = vertices.astype(np.float32)

    is_source = [np.zeros_like(boards[0], dtype=bool) for _ in range(len(boards))]

    m = None

    # now add transition faces
    for zind in range(len(boards) - 1):
        z = zs[zind]
        next_z = zs[zind + 1]
        for x in range(N - 1):
            for y in range(N - 1):
                old = boards[zind][x, y]
                new = boards[zind + 1][x, y]
                p4 = (x, y, next_z)
                p5 = (x + 1, y, next_z)
                p6 = (x + 1, y + 1, next_z)
                p7 = (x, y + 1, next_z)
                if (old == False) and (new == False):
                    # already dead
                    continue
                if (old == True) and (new == False):
                    # death
                    continue
                if (old == True) and (new == True):
                    # already alive
                    p0 = (x, y, z)
                    p1 = (x + 1, y, z)
                    p2 = (x + 1, y + 1, z)
                    p3 = (x, y + 1, z)
                if (old == False) and (new == True):
                    # birth
                    # find an alive neighbor from the previous board
                    for i in (-1, 0, 1):
                        for j in (-1, 0, 1):
                            if i == 0 and j == 0:
                                continue
                            if (
                                0 <= x + i < N - 1
                                and 0 <= y + j < N - 1
                                and boards[zind][x + i, y + j]
                            ):
                                oldx = x + i
                                oldy = y + j
                                is_source[zind][oldx, oldy] = True
                                break

                    p0 = (oldx, oldy, z)
                    p1 = (oldx + 1, oldy, z)
                    p2 = (oldx + 1, oldy + 1, z)
                    p3 = (oldx, oldy + 1, z)

                new_vertices = [
                    list(p0),
                    list(p1),
                    list(p2),
                    list(p3),
                    list(p4),
                    list(p5),
                    list(p6),
                    list(p7),
                ]

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

                if z == 0 and (x, y) in base_lego_points and old and new:
                    bot = zs[0]
                    base_lego = make_lego(
                        x,
                        y,
                        bot,
                        lego_height,
                        lego_width,
                        scale=1.05,
                    )
                    m = m - base_lego

    # add lego to the tops
    top = zs[-1]
    for x, y in top_lego_points:
        top_lego = make_lego(
            x,
            y,
            top,
            lego_height,
            lego_width,
            scale=1.0,
        )
        if m is not None:
            m = m + top_lego

    # m.vertices[:, 2] *= z_factor
    # m.vertices *= scale
    print(f"Is watertight: {m.is_watertight()}")
    m.to_stl("m.stl")
    t = trimesh.load_mesh("m.stl", process=False)

    z_factor = np.tan(np.radians(angle)) * np.sqrt(2)
    t.vertices[:, 2] *= z_factor
    t.vertices *= scale
    return t


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


if __name__ == "__main__":
    base = [
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
    ]

    top1 = [
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 1],
    ]

    top2 = [
        [1, 1, 1],
        [2, 1, 1],
        [2, 2, 1],
        [1, 2, 1],
    ]

    m1 = get_parallelipiped(base + top1)
    m2 = get_parallelipiped(base + top2)
    m = m1.union(m2)
    m.export("m.stl")
    m.is_watertight
