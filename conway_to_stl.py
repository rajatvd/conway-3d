import click
import os
import numpy as np
from conway3d import base_span, conway_step, boards_to_mesh_with_lego


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
            if base_init:
                splits[i] = [full_base, full_base] + s
            continue
        if base_each_slice:
            base = base_span(s, padding=2)
            splits[i] = [base, base] + s

    return splits


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
@click.option("--angle", type=float, default=55, help="Min overhang angle")
@click.option("--scale", type=float, default=5.0, help="Size of one cell in mm")
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
    scale,
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
    base_lego_points = []
    this_split_lego_points = []
    for i, split in enumerate(splits):
        if i < len(splits) - 1:
            next_split = splits[i + 1]
            this_split_lego_points = np.array(np.where(next_split[0] & next_split[1])).T
            top_lego_points = this_split_lego_points
        else:
            top_lego_points = []

        m = boards_to_mesh_with_lego(
            split,
            angle=angle,
            scale=scale,
            base_lego_points=base_lego_points,
            top_lego_points=top_lego_points,
            base_scale=0.2 if i == 0 and base_init else 1.0,
        )
        # m.save(f"{init_file}_{output}/{output}_{i}.stl")
        m.export(f"{init_file}_{output}/{output}_{i}.stl")
        base_lego_points = this_split_lego_points.copy()

    z_factor = np.tan(np.radians(angle)) * np.sqrt(2)
    print(f"{scale*z_factor:.4f} mm per generation")


if __name__ == "__main__":
    main()
