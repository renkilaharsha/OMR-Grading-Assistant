# Copyright 2022 Alexander L. Hayes
# Computer Vision - CSCI-B 657 - Spring 2022
# MIT License

"""
Tests for the Harris Corner Detection implementation.
"""

import pytest
import numpy as np
from harris.harris import HarrisCornerDetector
from harris.harris import Corner


def test_init_corner():
    crn = Corner(0, 1, 2.0)
    u, v = crn.coords
    assert u == 0
    assert v == 1
    assert crn.q == 2.0


def test_corner_distance_comparison():
    crn1 = Corner(0, 0, 0.0)
    crn2 = Corner(1, 1, 0.0)
    assert np.isclose(crn1.distance(crn2), 1.4142135623730951)


def test_comparing_corners():
    inp = [Corner(0, 0, 0.5), Corner(0, 0, 1.0), Corner(0, 0, 0.0)]
    out = [Corner(0, 0, 0.0), Corner(0, 0, 0.5), Corner(0, 0, 1.0)]
    assert out == sorted(inp)
