# kernels.py: Interesting kernels to use as templates.
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
A collection of interesting templates for the "row finding" problem.

Summary
-------

I didn't think of obvious names for these things, so the numbering scheme is
based on the order I tried them in. ``k3`` was the one I stopped with since
it seemed like the one that produced the fewest false positives in practice.

- k1: A period and a small vertical line.
- k2: The square period kernel.
- k3: A roundish period + full edge kernel.

Methods
-------

- ``make_k1()``: Return the k1 kernel.
- ``make_k2()``: Return the k2 kernel.
- ``make_k3()``: Return the k3 kernel.
- ``blur_kernel()``: Convolve an input kernel with a Gaussian blur filter.

Usage
-----

In practice, it seems like ``k3`` with a Gaussian blur is the best choice.

.. code-block:: python

    from template_matching.kernels import make_k3, blur_kernel

    kernel = blur_kernel(make_k3())
"""

from .filters import blur_filter
import numpy as np
from scipy.ndimage import convolve1d


def make_k1():
    """k1: A period and a small vertical line."""

    k1 = np.ones((8, 20))
    k1[:, 19:] = 0
    k1[3:6, 1:4] = 0
    k1[2, 2] = 0
    k1[3, 4] = 0
    k1 *= 255

    return k1


def make_k2():
    """k2: The square period kernel."""

    k2 = np.ones((7, 7))
    k2[2:5, 2:5] = 0
    k2 *= 255

    return k2


def make_k3():
    """k3: The roundish period + full edge kernel."""

    k3 = np.ones((36, 21))
    k3[1:35, 20] = 0
    k3[23:26, 2:5] = 0
    k3[26, 3] = 0
    k3[24, 5] = 0
    k3[22, 3] = 0
    k3[24, 1] = 0
    k3 *= 255

    return k3


def blur_kernel(kernel):
    """Convolve an input kernel with a Gaussian blur filter."""

    return convolve1d(
        convolve1d(
            kernel,
            blur_filter,
            mode='constant',
            cval=255.0,
            axis=1),
        blur_filter,
        mode='constant',
        cval=255.0,
        axis=0,
    )
