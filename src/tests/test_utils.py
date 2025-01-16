import pytest
from megafish.utils import natural_sort, get_tile_yx, get_round_cycle


def test_natural_sort():
    files = ["file10.txt", "file2.txt", "file1.txt"]
    sorted_files = natural_sort(files)
    assert sorted_files == ["file1.txt", "file2.txt", "file10.txt"]


def test_get_tile_yx():
    coords = get_tile_yx(2, 2, "row_right_down")
    assert len(coords) == 4


def test_get_round_cycle():
    rc = get_round_cycle(2, 3)
    assert len(rc) == 6
