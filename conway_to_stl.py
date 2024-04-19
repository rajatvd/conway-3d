import click
import os
import numpy as np
from conway3d import boards_to_mesh, base_span, conway_step


def conway_to_boards(
    init_file,
    num_generations,
    slices=None,
    slice_count=1,
    base_init=True,
    base_each_slice=False,
    padding=2,
    world_padding=20,
):
    """conway_to_stl.

    :param init: Path to the initial state file
    :param num_generations: Number of generations to simulate
    :param slices: Locations of slices (this overrides slice_count)
    :param slice_count: Number of slices to generate of equal length
    :param base_init: Whether to add a base to the initial state
    :param base_each_slice: Whether to add a base to each slice
    :param padding: Padding to add around the bases
    :param world_padding: Padding to add around the world
    """
    # Load the initial state
    with open(init_file, "r") as f:
        init_state = f.readlines()

    # Remove any trailing whitespace
    init_state = [line.strip() for line in init_state]

    # Get the dimensions of the initial state
    start = [row.strip() for row in init_state]
    start = np.array([list(row) for row in start if len(row) > 0])
    board = np.zeros_like(start, dtype=bool)
    board[start == "O"] = True

    # pad the board
    board = np.pad(board, world_padding)

    boards = [board]
    for _ in range(num_generations):
        board = conway_step(board)
        boards.append(board)

    full_base = base_span(boards, padding=padding)

    if slices is None and slice_count == 1:
        if base_init:
            boards = [full_base] + boards
        return [boards]

    if slices is not None:
        split_inds = slices
    else:
        slice_size = len(boards) // slice_count
        split_inds = list(range(0, len(boards), slice_size))
        split_inds[0] = 1

    splits = [
        boards[split_inds[i] - 1 : split_inds[i + 1]]
        for i in range(len(split_inds) - 1)
    ]

    for i, s in enumerate(splits):
        if i == 0:
            splits[i] = [full_base, full_base] + s
            continue
        if base_each_slice:
            base = base_span(s, padding=2)
            splits[i] = [base, base] + s

    return splits


def boards_to_stl(boards, filename, angle=55):
    """boards_to_stl.

    :param boards: List of boards to convert to STL
    :param filename: Filename to save the STL file
    :param angle: Min overhang angle
    """
    mesh = boards_to_mesh(boards, angle=angle)
    mesh.save(filename)


@click.command()
@click.argument("init_file", type=click.Path(exists=True))
@click.argument("num_generations", type=int)
@click.option(
    "--slices",
    type=list,
    default=None,
    help="Locations of slices (this overrides slice_count)",
)
@click.option(
    "--slice_count",
    type=int,
    default=1,
    help="Number of slices to generate of equal length",
)
@click.option(
    "--base_init", is_flag=True, help="Whether to add a base to the initial state"
)
@click.option(
    "--base_each_slice", is_flag=True, help="Whether to add a base to each slice"
)
@click.option("--padding", type=int, default=2, help="Padding to add around the bases")
@click.option("--output", type=str, default="output", help="Output filename")
@click.option(
    "--world_padding",
    type=int,
    default=20,
    help="Padding to add around the world",
)
@click.option("--angle", type=int, default=55, help="Min overhang angle")
def main(
    init_file,
    num_generations,
    slices,
    slice_count,
    base_init,
    base_each_slice,
    padding,
    output,
    world_padding,
    angle,
):
    """Main function for the CLI."""
    splits = conway_to_boards(
        init_file,
        num_generations,
        slices=slices,
        slice_count=slice_count,
        base_init=base_init,
        base_each_slice=base_each_slice,
        padding=padding,
        world_padding=world_padding,
    )
    # get init file name
    init_file = init_file.split("/")[-1].split(".")[0]
    os.makedirs(f"{init_file}_{output}", exist_ok=True)
    for i, split in enumerate(splits):
        boards_to_stl(
            split,
            f"{init_file}_{output}/{output}_{i}.stl",
            angle=angle,
        )


if __name__ == "__main__":
    main()
