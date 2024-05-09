import numpy as np
import trimesh
from oop3d import TrimeshThing


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

    fudge = 1e-5
    new_v = (new_v - new_v.mean(axis=0)) * (1 + fudge) + new_v.mean(axis=0)

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
