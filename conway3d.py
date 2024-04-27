import numpy as np
from functools import partial
import matplotlib.pyplot as plt
from stl import mesh


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
# given a sequence of boards, construct 3d model with z axis as the time axis
def boards_to_mesh(boards, angle=45):
    # angle is overhang angle in degrees
    # Create grid of vertices corresponding to cell corners
    N = len(boards[0]) + 1
    vertices = np.array(
        [[x, y, z] for x in range(N) for y in range(N) for z in range(len(boards))]
    )
    vertices = vertices.astype(np.float32)
    faces = []
    pti = partial(point_to_index, nx=N, ny=N, nz=len(boards))

    is_source = [np.zeros_like(boards[0], dtype=bool) for _ in range(len(boards))]

    # now add transition faces
    for x in range(N - 1):
        for y in range(N - 1):
            for z in range(len(boards) - 1):
                old = boards[z][x, y]
                new = boards[z + 1][x, y]
                p4 = pti(x, y, z + 1)
                p5 = pti(x + 1, y, z + 1)
                p6 = pti(x + 1, y + 1, z + 1)
                p7 = pti(x, y + 1, z + 1)
                if (old == False) and (new == False):
                    # already dead
                    continue
                if (old == True) and (new == False):
                    # death
                    continue
                if (old == True) and (new == True):
                    # already alive
                    p0 = pti(x, y, z)
                    p1 = pti(x + 1, y, z)
                    p2 = pti(x + 1, y + 1, z)
                    p3 = pti(x, y + 1, z)
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
                                and boards[z][x + i, y + j]
                            ):
                                oldx = x + i
                                oldy = y + j
                                is_source[z][oldx, oldy] = True
                                break

                    p0 = pti(oldx, oldy, z)
                    p1 = pti(oldx + 1, oldy, z)
                    p2 = pti(oldx + 1, oldy + 1, z)
                    p3 = pti(oldx, oldy + 1, z)

                # print("-------------------")
                # print("making parallelipied with vertices:")
                # ps = [p0, p1, p2, p3, p4, p5, p6, p7]
                # for p in ps:
                #     print(f"  {vertices[p]}")
                faces.extend(
                    [
                        [p0, p3, p1],
                        [p1, p3, p2],
                        [p0, p4, p7],
                        [p0, p7, p3],
                        [p4, p5, p6],
                        [p4, p6, p7],
                        [p5, p1, p2],
                        [p5, p2, p6],
                        [p2, p3, p6],
                        [p3, p7, p6],
                        [p0, p1, p5],
                        [p0, p5, p4],
                    ]
                )

    faces = np.array(faces)
    m = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))

    # find the z factor such that the max overhang angle is angle
    # tan(angle) = z_factor/sqrt{2}
    z_factor = np.tan(np.radians(angle)) * np.sqrt(2)
    zeros = np.where(vertices[:, 2] == 0)
    vertices[:, 2] *= z_factor
    zeros_after = np.where(vertices[:, 2] == 0)
    for i, f in enumerate(faces):
        for j in range(3):
            m.vectors[i][j] = vertices[f[j], :]
    return m


# %%
def boards_to_mesh_with_lego(
    boards,
    angle=45,
    base_lego_points=[],
    top_lego_points=[],
    lego_width=0.5,
    lego_height=0.7,
):
    # angle is overhang angle in degrees
    # Create grid of vertices corresponding to cell corners
    N = len(boards[0]) + 1
    vertices = np.array(
        [[x, y, z] for x in range(N) for y in range(N) for z in range(len(boards))]
    )
    vertices = vertices.astype(np.float32)
    faces = []
    pti = partial(point_to_index, nx=N, ny=N, nz=len(boards))

    is_source = [np.zeros_like(boards[0], dtype=bool) for _ in range(len(boards))]

    # now add transition faces
    for x in range(N - 1):
        for y in range(N - 1):
            for z in range(len(boards) - 1):
                old = boards[z][x, y]
                new = boards[z + 1][x, y]
                p4 = pti(x, y, z + 1)
                p5 = pti(x + 1, y, z + 1)
                p6 = pti(x + 1, y + 1, z + 1)
                p7 = pti(x, y + 1, z + 1)
                if (old == False) and (new == False):
                    # already dead
                    continue
                if (old == True) and (new == False):
                    # death
                    continue
                if (old == True) and (new == True):
                    # already alive
                    p0 = pti(x, y, z)
                    p1 = pti(x + 1, y, z)
                    p2 = pti(x + 1, y + 1, z)
                    p3 = pti(x, y + 1, z)
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
                                and boards[z][x + i, y + j]
                            ):
                                oldx = x + i
                                oldy = y + j
                                is_source[z][oldx, oldy] = True
                                break

                    p0 = pti(oldx, oldy, z)
                    p1 = pti(oldx + 1, oldy, z)
                    p2 = pti(oldx + 1, oldy + 1, z)
                    p3 = pti(oldx, oldy + 1, z)

                # print("-------------------")
                # print("making parallelipied with vertices:")
                # ps = [p0, p1, p2, p3, p4, p5, p6, p7]
                # for p in ps:
                #     print(f"  {vertices[p]}")
                faces.extend(
                    [
                        [p0, p3, p1],
                        [p1, p3, p2],
                        [p0, p4, p7],
                        [p0, p7, p3],
                        [p4, p5, p6],
                        [p4, p6, p7],
                        [p5, p1, p2],
                        [p5, p2, p6],
                        [p2, p3, p6],
                        [p3, p7, p6],
                        [p0, p1, p5],
                        [p0, p5, p4],
                    ]
                )

    # add lego to the tops
    top = len(boards) - 1.0
    border = (1 - lego_width) / 2.0
    lego_mid_height = lego_height - lego_width / 2.0
    for x, y in top_lego_points:
        new_vertices = [
            [x + border, y + border, top],
            [x + 1 - border, y + border, top],
            [x + 1 - border, y + 1 - border, top],
            [x + border, y + 1 - border, top],
            #
            [x + border, y + border, top + lego_mid_height],
            [x + 1 - border, y + border, top + lego_mid_height],
            [x + 1 - border, y + 1 - border, top + lego_mid_height],
            [x + border, y + 1 - border, top + lego_mid_height],
            #
            [x + 0.5, y + 0.5, top + lego_height],
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
            # [4, 5, 6],
            # [4, 6, 7],
            [4, 5, 8],
            [5, 6, 8],
            [6, 7, 8],
            [7, 4, 8],
        ]

        new_vertices = np.array(new_vertices)
        new_faces = list(np.array(new_faces) + len(vertices))

        vertices = np.vstack([vertices, new_vertices])
        faces.extend(new_faces)

    faces = np.array(faces)
    m = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))

    # find the z factor such that the max overhang angle is angle
    # tan(angle) = z_factor/sqrt{2}
    z_factor = np.tan(np.radians(angle)) * np.sqrt(2)
    zeros = np.where(vertices[:, 2] == 0)
    vertices[:, 2] *= z_factor
    zeros_after = np.where(vertices[:, 2] == 0)
    for i, f in enumerate(faces):
        for j in range(3):
            m.vectors[i][j] = vertices[f[j], :]
    return m


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
    boards = np.random.choice([False, True], (3, 10, 10))
    lego_points = [(3, 3), (3, 4), (4, 3), (4, 4)]
    m = boards_to_mesh_with_lego(boards, top_lego_points=lego_points)
    m.save("test.stl")
