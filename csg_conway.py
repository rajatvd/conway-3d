from netgen.csg import *
import numpy as np
from oop3d import CSGThing


def cross_product(v1, v2):
    return [
        v1[1] * v2[2] - v1[2] * v2[1],
        v1[2] * v2[0] - v1[0] * v2[2],
        v1[0] * v2[1] - v1[1] * v2[0],
    ]


def get_parallelipiped_csg(vertices):
    assert len(vertices) == 8, "Need 8 vertices to make a parallepiped"
    vertices = np.array(vertices)

    b = vertices[:4]
    t = vertices[4:]

    scale = 1
    b = (b - b.mean(axis=0)) * scale + b.mean(axis=0)
    t = (t - t.mean(axis=0)) * scale + t.mean(axis=0)

    fudge = 1e-2
    vertices = np.concatenate([b, t])
    vertices = (vertices - vertices.mean(axis=0)) * (1 + fudge) + vertices.mean(axis=0)
    vbot = vertices[0]
    vtop = vertices[6]

    left = Plane(
        Pnt(vbot[0], vbot[1], vbot[2]),
        Vec(cross_product(vertices[1] - vbot, vertices[5] - vbot)),
    )
    bot = Plane(
        Pnt(vbot[0], vbot[1], vbot[2]),
        Vec(cross_product(vertices[2] - vbot, vertices[1] - vbot)),
    )
    back = Plane(
        Pnt(vbot[0], vbot[1], vbot[2]),
        Vec(cross_product(vertices[4] - vbot, vertices[7] - vbot)),
    )

    right = Plane(
        Pnt(vtop[0], vtop[1], vtop[2]),
        Vec(cross_product(vertices[2] - vtop, vertices[3] - vtop)),
    )
    top = Plane(
        Pnt(vtop[0], vtop[1], vtop[2]),
        Vec(cross_product(vertices[7] - vtop, vertices[5] - vtop)),
    )
    front = Plane(
        Pnt(vtop[0], vtop[1], vtop[2]),
        Vec(cross_product(vertices[1] - vtop, vertices[2] - vtop)),
    )

    thing = left * right * front * back * bot * top

    # asdb = CSGeometry()
    # asdb.Add(thing)
    # mesh = asdb.GenerateMesh(maxh=0.1)
    # mesh.Export("parallepiped.stl", "STL Format")

    return CSGThing(thing)
    # return o3d.geometry.TriangleMesh(vertices, faces)


def make_lego_csg(
    x,
    y,
    z,
    lego_height,
    lego_width,
    scale=1.0,
):
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

    vertices = np.array(new_vertices)

    fudge = 1e-2
    vertices = (vertices - vertices.mean(axis=0)) * (1 + fudge) * scale + vertices.mean(
        axis=0
    )

    vbot = vertices[0]
    vtop = vertices[6]
    vcap = vertices[8]

    left = Plane(
        Pnt(vbot[0], vbot[1], vbot[2]),
        Vec(cross_product(vertices[1] - vbot, vertices[5] - vbot)),
    )
    bot = Plane(
        Pnt(vbot[0], vbot[1], vbot[2]),
        Vec(cross_product(vertices[2] - vbot, vertices[1] - vbot)),
    )
    back = Plane(
        Pnt(vbot[0], vbot[1], vbot[2]),
        Vec(cross_product(vertices[4] - vbot, vertices[7] - vbot)),
    )

    right = Plane(
        Pnt(vtop[0], vtop[1], vtop[2]),
        Vec(cross_product(vertices[2] - vtop, vertices[3] - vtop)),
    )
    front = Plane(
        Pnt(vtop[0], vtop[1], vtop[2]),
        Vec(cross_product(vertices[1] - vtop, vertices[2] - vtop)),
    )

    capleft = Plane(
        Pnt(vcap[0], vcap[1], vcap[2]),
        Vec(cross_product(vertices[4] - vcap, vertices[5] - vcap)),
    )
    capbot = Plane(
        Pnt(vcap[0], vcap[1], vcap[2]),
        Vec(cross_product(vertices[5] - vcap, vertices[6] - vcap)),
    )
    capright = Plane(
        Pnt(vcap[0], vcap[1], vcap[2]),
        Vec(cross_product(vertices[6] - vcap, vertices[7] - vcap)),
    )
    capfront = Plane(
        Pnt(vcap[0], vcap[1], vcap[2]),
        Vec(cross_product(vertices[7] - vcap, vertices[4] - vcap)),
    )

    thing = left * right * front * back * bot * capleft * capbot * capright * capfront

    return CSGThing(thing)
