# harris.py: Implementation of a Harris Corner Detector
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
Summary
-------

This implements the "Harris Corner Detection" method, described
in Chapter 4 of the Burger & Burge "Core Algorithms" book,
and expanded with a Java implementation in Appendix B.2,
starting on page 294.

Wikipedia highlights the five main steps:

1. Color to grayscale
2. Spatial derivative calculation
3. Structure tensor setup
4. Harris response calculation
5. Non-maximum suppression

Basic Usage
-----------

This is implemented as a class that implements a ``find_corners``
method. The ``find_corners`` method takes a grayscale image and
returns a list of ``Corner`` objects.

.. code-block:: python

    from harris.harris import HarrisCornerDetector
    from PIL import Image
    import numpy as np

    im = np.array(Image.open("docs/book_corners1.png"))

    hcd = HarrisCornerDetector(alpha=0.04, threshold=30_000)

    hcd.find_corners(im)

References
----------

- https://link.springer.com/book/10.1007/978-1-84800-195-4
- https://en.wikipedia.org/wiki/Harris_corner_detector

.. code-block:: bibtex

  @inbook{burger2009principles,
    author={Burger, Wilhelm and Burge, Mark J.},
    title={Corner Detection},
    booktitle={Principles of Digital Image Processing: Core Algorithms},
    chapter={4},
    pages={69-84},
    year={2009},
    publisher = {Springer Publishing Company, Incorporated},
    isbn = {978-1-84800-194-7},
    edition = {1},
  }
"""

from functools import total_ordering

import numpy as np
from PIL import Image
from PIL.JpegImagePlugin import JpegImageFile
from scipy.ndimage import convolve1d


def log_image(image, name: str):
    # Save a copy of a grayscale image.
    im = Image.fromarray(image.astype(np.uint8))
    im.save(name + ".png")


@total_ordering
class Corner:
    """A "Corner" has coordinates and a strength.

    Since they are point-like, you can calculate the
    distance between corners:

    >>> crn1 = Corner(0, 0)
    >>> crn2 = Corner(1, 1)
    >>> crn1.distance(crn2)
    1.4142135623730951

    Corners implement comparing in terms of their strength,
    which can be useful for getting the top strengths.

    >>> inp = [Corner(0, 0, 0.5), Corner(0, 0, 1.0), Corner(0, 0, 0.0)]
    >>> inp
    [Corner([0 0], q=0.5), Corner([0 0], q=1.0), Corner([0 0], q=0.0)]
    >>> sorted(inp)
    [Corner([0 0], q=0.0), Corner([0 0], q=0.5), Corner([0 0], q=1.0)]
    """

    def __init__(self, u: int, v: int, q: float = 0.0):
        self.coords = np.array([u, v])
        self.q = q

    def distance(self, other):
        assert type(self) == type(other)
        return np.linalg.norm(self.coords - other.coords)

    def __lt__(self, other):
        assert type(self) == type(other)
        return self.q < other.q

    def __eq__(self, other):
        assert type(self) == type(other)
        return self.q == other.q

    def __repr__(self):
        return self.__class__.__name__ + "(" + str(self.coords) + ", q=" + str(self.q) + ")"


class HarrisCornerDetector:
    """Harris Corner Detector object
    """

    def __init__(self, alpha=0.05, threshold=25000):
        """Initialize object

        Arguments:
            alpha: "Steering parameter" generally between 0.04 and 0.06
            threshold: "Response threshold" generally between 10,000 and 1,000,000
        """
        self.alpha = alpha
        self.threshold = threshold

    @staticmethod
    def is_local_max(Q, u, v):
        """Return True if Q[u, v] is a local maximum"""
        return Q[u, v] == np.max(Q[u-1:u+2, v-1:v+2])

    def collect_corners(self, Q, border=2):
        corner_list = []
        h, w = Q.shape

        for u in range(border, h - border):
            for v in range(border, w - border):
                q = Q[u, v]
                if q > self.threshold and self.is_local_max(Q, u, v):
                    corner_list.append(Corner(u, v, q))

        return corner_list

    def find_corners(self, image):
        """Apply the Harris detector to find corners in an image.

        Arguments:
            image: A grayscale image as a numpy array.

        Returns:
            A list of Corner objects.
        """

        imf = image.astype(np.float64)

        # "Make Derivatives"

        pfilt = np.array([0.223755, 0.552490, 0.223755])
        dfilt = np.array([0.453014, 0.0, -0.453014])
        bfilt = 1/64 * np.array([1.0, 6.0, 15.0, 20.0, 15.0, 6.0, 1.0])

        Ix = convolve1d(convolve1d(imf, pfilt, axis=1), dfilt, axis=1)
        Iy = convolve1d(convolve1d(imf, pfilt, axis=0), dfilt, axis=0)

        A = convolve1d(convolve1d(Ix ** 2, bfilt, axis=1), bfilt, axis=0)
        B = convolve1d(convolve1d(Iy ** 2, bfilt, axis=1), bfilt, axis=0)
        C = convolve1d(convolve1d(Ix * Iy, bfilt, axis=1), bfilt, axis=0)

        # "Make the Corner Response Function (CRF)"

        h, w = image.shape
        Q = np.zeros((h, w))

        for u in range(h):
            for v in range(w):

                a = A[u, v]
                b = B[u, v]
                c = C[u, v]

                det = (a * b) - (c * c)
                trace = a + b

                Q[u, v] = det - (self.alpha * (trace ** 2))

        # Collect the corners and perform non-maximal suppression.
        corners = self.collect_corners(Q)

        return corners

    def __repr__(self):
        _attributes = ["alpha", "threshold"]
        return (
            self.__class__.__name__
            + "("
            + ", ".join([attr + "=" + str(getattr(self, attr)) for attr in _attributes])
            + ")"
        )
