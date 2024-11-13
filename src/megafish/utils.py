import os
import shutil
import re
import numpy as np


def natural_sort(list_to_sort):
    """
    Sorts a list of strings in natural order, where numerical values are ordered as numbers,
    not as individual characters.

    Args:
        list_to_sort (list of str): The list of strings to be sorted.

    Returns:
        list of str: A new list sorted in natural order.

    Example:
        natural_sort(['file2.txt', 'file10.txt', 'file1.txt'])
        -> ['file1.txt', 'file2.txt', 'file10.txt']
    """
    def _natural_keys(text):
        # Split text into components: numbers are converted to integers, others remain as strings
        def _atoi(text):
            return int(text) if text.isdigit() else text
        return [_atoi(c) for c in re.split(r"(\d+)", text)]
    # Sort the list using the natural key function
    return sorted(list_to_sort, key=_natural_keys)


def get_tile_yx(n_tile_y, n_tile_x, scan_type):
    """
    Generates a list of (y, x) coordinates for tiles scanned according to a
    specified pattern.

    Args:
        n_tile_y (int): The number of tiles in the y direction.
        n_tile_x (int): The number of tiles in the x direction.
        scan_type (str): The scanning pattern type, e.g., "snake_up_right" or
            "snake_right_down".

    Returns:
        list of tuple: A list of (y, x) coordinates for each tile.

    Raises:
        ValueError: If the scan_type is not supported.
    """
    # Horizontal snake
    if scan_type == "snake_right_down":
        tile_x = [np.arange(n_tile_x) if i % 2 == 0 else np.arange(
            n_tile_x)[::-1] for i in range(n_tile_y)]
        tile_y = np.repeat(np.arange(n_tile_y), n_tile_x)
    elif scan_type == "snake_left_down":
        tile_x = [np.arange(n_tile_x)[::-1] if i % 2 ==
                  0 else np.arange(n_tile_x) for i in range(n_tile_y)]
        tile_y = np.repeat(np.arange(n_tile_y), n_tile_x)
    elif scan_type == "snake_right_up":
        tile_x = [np.arange(n_tile_x) if i % 2 == 0 else np.arange(
            n_tile_x)[::-1] for i in range(n_tile_y)]
        tile_y = np.repeat(np.arange(n_tile_y)[::-1], n_tile_x)
    elif scan_type == "snake_left_up":
        tile_x = [np.arange(n_tile_x)[::-1] if i % 2 ==
                  0 else np.arange(n_tile_x) for i in range(n_tile_y)]
        tile_y = np.repeat(np.arange(n_tile_y)[::-1], n_tile_x)

    # Vertical snake
    elif scan_type == "snake_up_right":
        tile_y = [np.arange(n_tile_y)[::-1] if i % 2 ==
                  0 else np.arange(n_tile_y) for i in range(n_tile_x)]
        tile_x = np.repeat(np.arange(n_tile_x), n_tile_y)
    elif scan_type == "snake_up_left":
        tile_y = [np.arange(n_tile_y)[::-1] if i % 2 ==
                  0 else np.arange(n_tile_y) for i in range(n_tile_x)]
        tile_x = np.repeat(np.arange(n_tile_x)[::-1], n_tile_y)
    elif scan_type == "snake_down_right":
        tile_y = [np.arange(n_tile_y) if i % 2 == 0 else np.arange(
            n_tile_y)[::-1] for i in range(n_tile_x)]
        tile_x = np.repeat(np.arange(n_tile_x), n_tile_y)
    elif scan_type == "snake_down_left":
        tile_y = [np.arange(n_tile_y) if i % 2 == 0 else np.arange(
            n_tile_y)[::-1] for i in range(n_tile_x)]
        tile_x = np.repeat(np.arange(n_tile_x)[::-1], n_tile_y)

    # Row scan
    elif scan_type == "row_right_down":
        tile_x = np.tile(np.arange(n_tile_x), n_tile_y)
        tile_y = np.repeat(np.arange(n_tile_y), n_tile_x)
    elif scan_type == "row_left_down":
        tile_x = np.tile(np.arange(n_tile_x)[::-1], n_tile_y)
        tile_y = np.repeat(np.arange(n_tile_y), n_tile_x)
    elif scan_type == "row_right_up":
        tile_x = np.tile(np.arange(n_tile_x), n_tile_y)
        tile_y = np.repeat(np.arange(n_tile_y)[::-1], n_tile_x)
    elif scan_type == "row_left_up":
        tile_x = np.tile(np.arange(n_tile_x)[::-1], n_tile_y)
        tile_y = np.repeat(np.arange(n_tile_y)[::-1], n_tile_x)

    # Column scan
    elif scan_type == "column_down_right":
        tile_y = np.tile(np.arange(n_tile_y), n_tile_x)
        tile_x = np.repeat(np.arange(n_tile_x), n_tile_y)
    elif scan_type == "column_down_left":
        tile_y = np.tile(np.arange(n_tile_y), n_tile_x)
        tile_x = np.repeat(np.arange(n_tile_x)[::-1], n_tile_y)
    elif scan_type == "column_up_right":
        tile_y = np.tile(np.arange(n_tile_y)[::-1], n_tile_x)
        tile_x = np.repeat(np.arange(n_tile_x), n_tile_y)
    elif scan_type == "column_up_left":
        tile_y = np.tile(np.arange(n_tile_y)[::-1], n_tile_x)
        tile_x = np.repeat(np.arange(n_tile_x)[::-1], n_tile_y)
    else:
        raise ValueError("Unsupported scan_type")

    # Flatten the lists if they were created with list comprehensions
    if isinstance(tile_y, list):
        tile_y = np.concatenate(tile_y)
    if isinstance(tile_x, list):
        tile_x = np.concatenate(tile_x)

    return list(zip(tile_y, tile_x))


def get_round_cycle(n_round, n_cycle):
    """
    Generates a list of (round, cycle) pairs for each round and cycle.

    Args:
        n_round (int): The number of rounds.
        n_cycle (int): The number of cycles.

    Returns:
        list of tuple: A list of (round, cycle) pairs for each round and cycle.
    """
    return [(round_, cycle) for round_ in range(n_round) for cycle in range(n_cycle)]


def copy_groups(src_dir, zarr_path, groups):
    for group in groups:
        print("Copying: " + group)
        sample_name = zarr_path.split("/")[-1].split(".")[0]
        src_path = os.path.join(src_dir, sample_name + ".zarr", group)
        dst_path = os.path.join(zarr_path, group)

        shutil.copytree(src_path, dst_path)
