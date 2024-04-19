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

    # add death caps
    # for x in range(N - 1):
    #     for y in range(N - 1):
    #         for z in range(len(boards) - 1):
    #             if is_source[z][x, y]:
    #                 continue
    #             old = boards[z][x, y]
    #             new = boards[z + 1][x, y]
    #             if (old == True) and (new == False):
    #                 p1 = pti(x, y, z)
    #                 p2 = pti(x + 1, y, z)
    #                 p3 = pti(x, y + 1, z)
    #                 p4 = pti(x + 1, y + 1, z)
    #                 faces.extend(
    #                     [
    #                         [p1, p2, p3],
    #                         [p2, p4, p3],
    #                     ]
    #                 )

    # add the faces on base
    # for x in range(N - 1):
    #     for y in range(N - 1):
    #         for z in [0]:
    #             if not boards[z][x, y]:
    #                 continue
    #             # Create two triangles for each face
    #             p0 = pti(x, y, z)
    #             p1 = pti(x + 1, y, z)
    #             p2 = pti(x + 1, y + 1, z)
    #             p3 = pti(x, y + 1, z)
    #             faces.extend(
    #                 [
    #                     [p0, p3, p1],
    #                     [p1, p3, p2],
    #                     [p0, p1, p3],
    #                     [p1, p2, p3],
    #                 ]
    #             )

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


# %%
# Example usage
def glider():
    N = 50
    c = 10
    steps = 8
    width = 5
    # start with a glider in the center
    board = np.zeros((N, N), dtype=bool)
    board[c - 4, c - 3] = True
    board[c - 3, c - 2] = True
    board[c - 2, c - 4 : c - 1] = True

    boards = [board]
    for _ in range(steps):
        board = conway_step(board)
        boards.append(board)

    # add a base board to make the model more stable
    base = np.zeros((N, N), dtype=bool)
    base[c - width : c, c - width : c] = True
    boards = [base, base] + boards
    return boards


def methuselah(count, slices=None):
    N = 50
    c = 10
    steps = 150
    # start with a die-hard methuselah in the center
    board = np.zeros((N, N), dtype=bool)
    board[c + 4, c + 4 : c + 7] = True
    board[c + 2, c + 5] = True

    board[c + 4, c] = True
    board[c + 3, c] = True
    board[c + 3, c - 1] = True

    boards = [board]
    for _ in range(steps):
        board = conway_step(board)
        boards.append(board)

    if slices is not None:
        split_inds = slices
    else:
        split_inds = list(range(0, len(boards), count))
        split_inds[0] = 1
    print(split_inds)
    splits = [
        boards[split_inds[i] - 1 : split_inds[i + 1]]
        for i in range(len(split_inds) - 1)
    ]
    # add bases to each of the splits
    full_base = base_span(boards, padding=3)
    for i, s in enumerate(splits):
        if i == 0:
            splits[i] = [full_base, full_base] + s
            continue
        base = base_span(s, padding=2)
        # splits[i] = [base, base] + s
    return splits


def glider_gun(count):
    N = 50
    c = 5
    steps = 90
    # start with a glider gun in the center
    board = np.zeros((N, N), dtype=bool)
    gun = np.array(
        [
            "0000000000000000000000000100000000000",
            "0000000000000000000000010100000000000",
            "0000000000000110000001100000000000011",
            "0000000000001000100001100000000000011",
            "0110000000010000010001100000000000000",
            "0110000000010001011000010100000000000",
            "0000000000010000010000000100000000000",
            "0000000000001000100000000000000000000",
            "0000000000000110000000000000000000000",
        ]
    )
    gun = np.array([list(row) for row in gun])
    board[c : c + gun.shape[0], c : c + gun.shape[1]] = gun == "1"

    boards = [board]
    for _ in range(steps):
        board = conway_step(board)
        boards.append(board)

    # add a base board to make the model more stable
    base = np.zeros((N, N), dtype=bool)
    base[2:30, 2:45] = True
    boards = [base, base] + boards[10:]
    return [boards]


def lwss(count, **kwargs):
    N = 50
    c = 40
    steps = 30
    # start with a lightweight spaceship in the center
    board = np.zeros((N, N), dtype=bool)
    lwss = np.array(
        [
            "11110",
            "10001",
            "10000",
            "01001",
        ]
    )
    lwss = np.array([list(row) for row in lwss])
    board[c : c + lwss.shape[0], c : c + lwss.shape[1]] = lwss == "1"

    boards = [board]
    for _ in range(steps):
        board = conway_step(board)
        boards.append(board)

    # add a base board to make the model more stable
    base = np.zeros((N, N), dtype=bool)
    base[38:45, 22:48] = True
    boards = [base, base] + boards
    return [boards]


# %%
def infinite_growth(count):
    N = 50
    c = 20
    steps = 35
    board = np.zeros((N, N), dtype=bool)
    start = np.array(
        [
            "00000010",
            "00001011",
            "00001010",
            "00001000",
            "00100000",
            "10100000",
        ]
    )
    start = np.array([list(row) for row in start])
    board[c : c + start.shape[0], c : c + start.shape[1]] = start == "1"

    boards = [board]
    for _ in range(steps):
        board = conway_step(board)
        boards.append(board)

    # add a base board to make the model more stable
    base = np.zeros((N, N), dtype=bool)
    base[8:32, 15:34] = True
    boards = [base, base] + boards
    return [boards]


def infinite_growth_inverse(count):
    N = 50
    c = 20
    steps = 80
    board = np.zeros((N, N), dtype=bool)
    start = np.array(
        [
            "00000010",
            "00001011",
            "00001010",
            "00001000",
            "00100000",
            "10100000",
        ]
    )
    start = np.array([list(row) for row in start])
    board[c : c + start.shape[0], c : c + start.shape[1]] = start == "1"

    boards = [board]
    for _ in range(steps):
        board = conway_step(board)
        boards.append(board)

    # add a base board to make the model more stable
    # base = np.zeros((N, N), dtype=bool)
    # base[8:32, 15:34] = True
    boards = boards
    return [boards]


def four_guns(count, **kwargs):
    N = 100
    c = 5
    steps = 50
    # start with a glider gun in the center
    board = np.zeros((N, N), dtype=bool)
    gun = np.array(
        [
            "0000000000000000000000000100000000000",
            "0000000000000000000000010100000000000",
            "0000000000000110000001100000000000011",
            "0000000000001000100001100000000000011",
            "0110000000010000010001100000000000000",
            "0110000000010001011000010100000000000",
            "0000000000010000010000000100000000000",
            "0000000000001000100000000000000000000",
            "0000000000000110000000000000000000000",
        ]
    )
    gun = np.array([list(row) for row in gun])
    # pad the smaller axis with 0s to make it square by adding 0s to the right or top
    if gun.shape[0] < gun.shape[1]:
        gun = np.concatenate(
            (np.zeros((gun.shape[1] - gun.shape[0], gun.shape[1])), gun), axis=0
        )
    else:
        gun = np.concatenate(
            (np.zeros((gun.shape[0], gun.shape[0] - gun.shape[1])), gun), axis=1
        )

    # make 4 rotated guns such that all gliders hit the same point
    left_guns = gun.copy()

    gun = np.rot90(gun)
    left_guns = np.concatenate((left_guns, gun), axis=0)

    gun = np.rot90(gun)
    right_guns = gun.copy()

    gun = np.rot90(gun)
    right_guns = np.concatenate((gun, right_guns), axis=0)

    all_guns = np.concatenate((left_guns, right_guns), axis=1)

    board[c : c + all_guns.shape[0], c : c + all_guns.shape[1]] = all_guns == "1"

    boards = [board]
    for _ in range(steps):
        board = conway_step(board)
        boards.append(board)

    # add a base board to make the model more stable
    base = base_span(boards, padding=2)
    boards = [base, base] + boards

    return [boards]


def four_glider_collision(count, **kwargs):
    N = 100
    c = 5
    steps = 50
    # start with a glider gun in the center
    board = np.zeros((N, N), dtype=bool)

    # 4 gliders
    start = np.array([])
    start = np.array([list(row) for row in start])

    board[c : c + start.shape[0], c : c + start.shape[1]] = start == "1"
    boards = [board]
    for _ in range(steps):
        board = conway_step(board)
        boards.append(board)

    # add a base board to make the model more stable
    base = base_span(boards, padding=2)
    boards = [base, base] + boards
    return [boards]


def chaotic_growth(count, **kwargs):
    N = 100
    c = 50
    steps = 50
    board = np.zeros((N, N), dtype=bool)

    start = np.array(
        [
            "001",
            "011",
            "110",
            "100",
        ]
    )

    start = np.array([list(row) for row in start])
    board[c : c + start.shape[0], c : c + start.shape[1]] = start == "1"
    boards = [board]
    for _ in range(steps):
        board = conway_step(board)
        boards.append(board)

    # add a base board to make the model more stable
    base = base_span(boards, padding=2)
    boards = [base, base] + boards
    return [boards]


def four_to_five(count, **kwargs):
    N = 200
    c = 100
    steps = 150
    board = np.zeros((N, N), dtype=bool)

    with open("4g-5g.txt") as f:
        start = f.readlines()
    start = [row.strip() for row in start]
    start = np.array([list(row) for row in start if len(row) > 0])

    board[c : c + start.shape[0], c : c + start.shape[1]] = start == "O"
    boards = [board]
    for _ in range(steps):
        board = conway_step(board)
        boards.append(board)

    base = base_span(boards, padding=2)
    splits = [[base, base] + boards[20:53], boards[52:86], boards[85:125], boards[124:]]
    # add a base board to make the model more stable
    return splits


if __name__ == "__main__":
    # example sequence of boards with one cell moving across the board
    # boards = [np.zeros((N, N), dtype=bool) for _ in range(steps)]
    # for i in range(steps):
    #     boards[i][0, min(i, N - 1)] = True
    thingys = {
        "glider": glider,
        "methuselah": methuselah,
        "glider_gun": glider_gun,
        "lwss": lwss,
        "infinite_growth": infinite_growth,
        "infinite_growth_inverse": infinite_growth_inverse,
        "four_guns": four_guns,
        "four_glider_collision": four_glider_collision,
        "chaotic_growth": chaotic_growth,
        "four_to_five": four_to_five,
    }

    name = "four_to_five"

    count = 45
    # slices = [1, 17, 35, 59, 80, 108, 150]
    slices = None
    splits = thingys[name](count, slices=slices)

    plt.ion()
    fig, ax = plt.subplots()
    k = 1
    for i, boards in enumerate(splits):
        for b in boards:
            ax.clear()
            ax.set_title(f"split {i}, iter {k}")
            ax.imshow(b, cmap="binary")
            plt.pause(0.1)
            k += 1
    plt.ioff()
    plt.show()
    # Construct 3D mesh

    angle = 55
    for i, s in enumerate(splits):
        m = boards_to_mesh(s, angle=angle)
        m.check(exact=True)
        m.save(f"{name}{i}_angle{angle}_count{count}_num_splits{len(splits)}.stl")


# %%

# import pymesh

# pm = pymesh.load_mesh(
#     "methuselah4_angle55_count45_slices[1, 17, 35, 59, 80, 108, 150].stl"
# )

# # pm, info = pymesh.remove_isolated_vertices(pm)
# # pm, info = pymesh.remove_duplicated_faces(pm)
# # pm, info = pymesh.remove_degenerated_triangles(pm)
# # pymesh.save_mesh(
# #     "methuselah4_angle55_count45_slices[1, 17, 35, 59, 80, 108, 150]_clean.stl", pm
# # )
