import numpy as np
from conway_to_stl import conway_to_boards
from tqdm import tqdm
import matplotlib.pyplot as plt
import imageio

# %%

false_value = 0.1
true_value = 0.8


def make_gif(
    init_file,
    num_generations=135,
    output="conway.gif",
    expansion=4,
):
    boards = conway_to_boards(
        init_file,
        num_generations=num_generations,
        slices=None,
        slice_count=1,
        world_padding=9,
    )[0]

    expanded_boards = []
    for board in boards:
        expanded_board = np.zeros(
            (expansion * board.shape[0], expansion * board.shape[1]),
            dtype=bool,
        )
        for i in range(board.shape[0]):
            for j in range(board.shape[1]):
                expanded_board[
                    i * expansion : (i + 1) * expansion,
                    j * expansion : (j + 1) * expansion,
                ] = board[i, j]
        expanded_boards.append(expanded_board)

    trues = np.where(expanded_boards)
    boards_vals = np.full_like(expanded_boards, false_value, dtype=float)
    boards_vals[trues] = true_value

    images = []
    for i, board in tqdm(enumerate(boards_vals)):
        plt.figure(figsize=(35, 40))
        cmap = plt.cm.get_cmap("Greens")
        plt.imshow(board, cmap=cmap, vmin=0, vmax=1)
        plt.axis("off")
        plt.savefig("temp.png", bbox_inches="tight", pad_inches=0)
        plt.close()
        image = imageio.imread("temp.png")
        images.append(image)

    imageio.mimsave(output, images)


# %%
make_gif(
    "die-hard.txt",
    num_generations=135,
    output="unsmoothed-die-hard.mp4",
    expansion=30,
)


# %%
def make_gif_smooth(
    init_file,
    num_generations=135,
    expansion=4,
    output="conway.gif",
):
    boards = conway_to_boards(
        init_file,
        num_generations=num_generations,
        slices=None,
        slice_count=1,
        world_padding=9,
    )[0]

    # expand each tile by expansion in both dimensions
    expanded_boards = []
    for board in boards:
        expanded_board = np.zeros(
            (expansion * board.shape[0], expansion * board.shape[1]),
            dtype=bool,
        )
        for i in range(board.shape[0]):
            for j in range(board.shape[1]):
                expanded_board[
                    i * expansion : (i + 1) * expansion,
                    j * expansion : (j + 1) * expansion,
                ] = board[i, j]
        expanded_boards.append(expanded_board)

    # smoothen transitions between boards by moving across expansion
    smooth_boards = np.zeros_like(expanded_boards[0], dtype=bool)[None, ...]
    for bigz in range(len(boards) - 1):
        transitions = np.zeros((expansion, *expanded_boards[0].shape), dtype=bool)
        transitions[0] = expanded_boards[bigz]

        for x in range(boards[0].shape[0]):
            for y in range(boards[0].shape[1]):
                old = boards[bigz][x, y]
                new = boards[bigz + 1][x, y]
                if (old == False) and (new == False):
                    # already dead
                    continue
                if (old == True) and (new == False):
                    # death
                    continue
                if (old == True) and (new == True):
                    # already alive
                    for z in range(expansion):
                        transitions[
                            z,
                            x * expansion : (x + 1) * expansion,
                            y * expansion : (y + 1) * expansion,
                        ] = True

                if (old == False) and (new == True):
                    # birth

                    oldx = x
                    oldy = y
                    # find an alive neighbor from the previous board
                    for ii in (-1, 0, 1):
                        for jj in (-1, 0, 1):
                            if ii == 0 and jj == 0:
                                continue
                            if (
                                0 <= x + ii < boards[0].shape[0] - 1
                                and 0 <= y + jj < boards[0].shape[1] - 1
                                and boards[bigz][x + ii, y + jj]
                            ):
                                oldx = x + ii
                                oldy = y + jj
                                break

                    # transition from old to new
                    xdelta = x - oldx
                    ydelta = y - oldy

                    assert xdelta != 0 or ydelta != 0, "No neighbor found for birth"

                    for z in range(expansion):
                        transitions[
                            z,
                            oldx * expansion
                            + xdelta * z : (oldx + 1) * expansion
                            + xdelta * z,
                            oldy * expansion
                            + ydelta * z : (oldy + 1) * expansion
                            + ydelta * z,
                        ] = True

        smooth_boards = np.append(smooth_boards, transitions, axis=0)

    trues = np.where(smooth_boards)
    smooth_boards_vals = np.full_like(smooth_boards, false_value, dtype=float)
    smooth_boards_vals[trues] = true_value

    images = []
    for i, board in tqdm(enumerate(smooth_boards_vals)):
        plt.figure(figsize=(7, 7))
        # set colormap to a lightish green and a bluish white background
        cmap = plt.cm.get_cmap("Greens")
        # cmap.set_bad(color="white")
        plt.imshow(board, cmap=cmap, vmin=0, vmax=1)
        plt.axis("off")
        # plt.grid(True)
        plt.savefig("temp.png", bbox_inches="tight", pad_inches=0)
        plt.close()
        image = imageio.imread("temp.png")
        images.append(image)

    imageio.mimsave(output, images)


# %%
ex = 5
make_gif_smooth(
    "die-hard.txt",
    expansion=ex,
    num_generations=135,
    output=f"smoothed-die-hard_expansion{ex}.mp4",
)
