# Copyright 2022 Alexander L. Hayes
# Computer Vision - CSCI-B 657 - Spring 2022
# MIT License

"""
Utils for creating crops of a bigger image, and
splitting thsoe crops into meaningful regions.
"""

import numpy as np


def make_crop(image, x, y):
    """Make a fixed-size crop starting from the top-left corner of an image.

    Arguments:
        image: PIL image where the `crop` method is defined
        x: Top-left `x` coordinate
        y: Top-left `y` coordinate

    Returns:
        (44, 400) numpy array at the location
    """
    return np.array(image.crop((x, y, x + 400, y + 44)))


def split_crop(crop):
    """Split out the three *interesting* regions in a crop.

    - 0: Where a "written set of characters" occur
    - 1: Where the "problem number" occurs
    - 2: Where the boxes for A/B/C/D/E occur
    """
    return (
        crop[:, 0:70],
        crop[:, 71:115],
        crop[:, 116:],
    )


def split_right(crop):
    """Look for the boxes of letters."""
    return (
        crop[:, 0:56],
        crop[:, 57:113],
        crop[:, 114:170],
        crop[:, 171:227],
        crop[:, 228:285],
    )


def onehot_to_label(vector):
    """Convert a 5-vector of True/False values to a string of A/B/C/D/E"""
    _table = {
        0: "A",
        1: "B",
        2: "C",
        3: "D",
        4: "E",
    }
    out = ""
    for i, entry in enumerate(vector):
        if entry:
            out += _table[i]
    return out
