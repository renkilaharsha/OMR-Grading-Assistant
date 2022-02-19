# Computer Vision - CSCI-B 657 - Spring 2022

# MIT License

# Copyright © 2022 Alexander L. Hayes (hayesall)

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

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
