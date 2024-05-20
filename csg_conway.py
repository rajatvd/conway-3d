from netgen.csg import *
import numpy as np
from oop3d import CSGThing


def cross_product(v1, v2):
    return [
        v1[1] * v2[2] - v1[2] * v2[1],
        v1[2] * v2[0] - v1[0] * v2[2],
        v1[0] * v2[1] - v1[1] * v2[0],
    ]


# %%
LEFT = 0
RIGHT = 1
FRONT = 2
BACK = 3
TOP = 4
BOT = 5


# %%
def make_hexahedron_csg(
    vertices,
    faces=[LEFT, RIGHT, FRONT, BACK, TOP, BOT],
):
    vbot = vertices[0]
    vtop = vertices[6]

    all_faces = [None for _ in range(6)]
    all_faces[LEFT] = Plane(
        Pnt(vbot[0], vbot[1], vbot[2]),
        Vec(cross_product(vertices[1] - vbot, vertices[5] - vbot)),
    )
    all_faces[BOT] = Plane(
        Pnt(vbot[0], vbot[1], vbot[2]),
        Vec(cross_product(vertices[2] - vbot, vertices[1] - vbot)),
    )
    all_faces[BACK] = Plane(
        Pnt(vbot[0], vbot[1], vbot[2]),
        Vec(cross_product(vertices[4] - vbot, vertices[7] - vbot)),
    )

    all_faces[RIGHT] = Plane(
        Pnt(vtop[0], vtop[1], vtop[2]),
        Vec(cross_product(vertices[2] - vtop, vertices[3] - vtop)),
    )
    all_faces[TOP] = Plane(
        Pnt(vtop[0], vtop[1], vtop[2]),
        Vec(cross_product(vertices[7] - vtop, vertices[5] - vtop)),
    )
    all_faces[FRONT] = Plane(
        Pnt(vtop[0], vtop[1], vtop[2]),
        Vec(cross_product(vertices[1] - vtop, vertices[2] - vtop)),
    )

    thing = None
    for i in faces:
        if thing is None:
            thing = all_faces[i]
        else:
            thing = thing * all_faces[i]

    return CSGThing(thing)


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

    hex = make_hexahedron_csg(vertices)
    return hex


def make_lego_csg(
    x,
    y,
    z,
    lego_height,
    lego_width,
    scale=1.0,
    sign=1,
    bottom=True,
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

    if not bottom:
        left = Plane(
            Pnt(vbot[0], vbot[1], vbot[2]),
            sign * Vec(cross_product(vertices[1] - vbot, vertices[5] - vbot)),
        )
        bot = Plane(
            Pnt(vbot[0], vbot[1], vbot[2]),
            sign * Vec(cross_product(vertices[2] - vbot, vertices[1] - vbot)),
        )
        back = Plane(
            Pnt(vbot[0], vbot[1], vbot[2]),
            sign * Vec(cross_product(vertices[4] - vbot, vertices[7] - vbot)),
        )

        right = Plane(
            Pnt(vtop[0], vtop[1], vtop[2]),
            sign * Vec(cross_product(vertices[2] - vtop, vertices[3] - vtop)),
        )
        front = Plane(
            Pnt(vtop[0], vtop[1], vtop[2]),
            sign * Vec(cross_product(vertices[1] - vtop, vertices[2] - vtop)),
        )

        capleft = Plane(
            Pnt(vcap[0], vcap[1], vcap[2]),
            sign * Vec(cross_product(vertices[4] - vcap, vertices[5] - vcap)),
        )
        capbot = Plane(
            Pnt(vcap[0], vcap[1], vcap[2]),
            sign * Vec(cross_product(vertices[5] - vcap, vertices[6] - vcap)),
        )
        capright = Plane(
            Pnt(vcap[0], vcap[1], vcap[2]),
            sign * Vec(cross_product(vertices[6] - vcap, vertices[7] - vcap)),
        )
        capfront = Plane(
            Pnt(vcap[0], vcap[1], vcap[2]),
            sign * Vec(cross_product(vertices[7] - vcap, vertices[4] - vcap)),
        )

        if sign == -1:
            thing = (
                left
                + right
                + front
                + back
                + bot
                + capleft
                + capbot
                + capright
                + capfront
            )
        else:
            thing = (
                left
                * right
                * front
                * back
                * bot
                * capleft
                * capbot
                * capright
                * capfront
            )

    else:
        left = Plane(
            Pnt(vbot[0], vbot[1], vbot[2]),
            Vec(cross_product(vertices[1] - vbot, vertices[5] - vbot)),
        )
        back = Plane(
            Pnt(vbot[0], vbot[1], vbot[2]),
            sign * Vec(cross_product(vertices[4] - vbot, vertices[7] - vbot)),
        )

        right = Plane(
            Pnt(vtop[0], vtop[1], vtop[2]),
            sign * Vec(cross_product(vertices[2] - vtop, vertices[3] - vtop)),
        )
        front = Plane(
            Pnt(vtop[0], vtop[1], vtop[2]),
            sign * Vec(cross_product(vertices[1] - vtop, vertices[2] - vtop)),
        )
        vothertop = vertices[4]
        capleft = Plane(
            Pnt(vothertop[0], vothertop[1], vothertop[2]),
            sign * Vec(cross_product(vertices[4] - vcap, vertices[5] - vcap)),
        )
        capbot = Plane(
            Pnt(vtop[0], vtop[1], vtop[2]),
            sign * Vec(cross_product(vertices[5] - vcap, vertices[6] - vcap)),
        )
        capright = Plane(
            Pnt(vtop[0], vtop[1], vtop[2]),
            sign * Vec(cross_product(vertices[6] - vcap, vertices[7] - vcap)),
        )
        capfront = Plane(
            Pnt(vothertop[0], vothertop[1], vothertop[2]),
            sign * Vec(cross_product(vertices[7] - vcap, vertices[4] - vcap)),
        )

        if sign == -1:
            thing = left + right + front + back + capleft + capbot + capright + capfront
        else:
            thing = left * right * front * back * capleft * capbot * capright * capfront

    return CSGThing(thing)


# %%
def get_thin_transition_csg(vertices):
    assert len(vertices) == 8, "Need 8 vertices to make a transition"
    vertices = np.array(vertices)

    b = vertices[:4]
    t = vertices[4:]

    scale = 1.05
    b = (b - b.mean(axis=0)) * scale + b.mean(axis=0)
    t = (t - t.mean(axis=0)) * scale + t.mean(axis=0)

    b_to_t_vector = t.mean(axis=0) - b.mean(axis=0)

    fudge = 1e-2
    vertices = np.concatenate([b, t])
    vertices = (vertices - vertices.mean(axis=0)) * (1 + fudge) + vertices.mean(axis=0)

    inner_scale = 0.5
    inner_beam_vertices = (vertices - vertices.mean(axis=0)) * inner_scale
    inner_beam_vertices[:4] -= b_to_t_vector[None, :] * 0.1
    inner_beam_vertices[4:] += b_to_t_vector[None, :] * 0.1
    inner_vertices = inner_beam_vertices * 0.5 + vertices.mean(axis=0)
    inner_beam_vertices = inner_beam_vertices + vertices.mean(axis=0)

    lower_vertices = np.concatenate([vertices[:4], inner_vertices[:4]])

    upper_vertices = np.concatenate([inner_vertices[4:], vertices[4:]])

    inner = make_hexahedron_csg(inner_beam_vertices, faces=[LEFT, RIGHT, FRONT, BACK])
    upper = make_hexahedron_csg(upper_vertices, faces=[LEFT, RIGHT, FRONT, BACK, TOP])
    lower = make_hexahedron_csg(lower_vertices, faces=[LEFT, RIGHT, FRONT, BACK, BOT])

    everything = make_hexahedron_csg(vertices)

    thing = (inner + upper + lower) * everything
    # inner.to_stl("inner.stl")
    # upper.to_stl("upper.stl")
    # lower.to_stl("lower.stl")
    # thing.to_stl("thing.stl")

    return thing
