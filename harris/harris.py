# Copyright 2022 Alexander L. Hayes
# Computer Vision - CSCI-B 657 - Spring 2022
# MIT License

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
from math import inf

import numpy as np
from PIL import Image
from PIL.JpegImagePlugin import JpegImageFile
from scipy.ndimage import convolve1d


def log_image(image, name: str):
    # Save a copy of a grayscale image.
    im = Image.fromarray(image.astype(np.uint8))
    im.save(name + ".jpg")


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

    - Images must be grayscale.
    """

    def __init__(self, alpha=0.05, threshold=20000):
        """Initialize object

        Arguments:
            alpha: "Steering parameter" generally between 0.04 and 0.06
            threshold: "Response threshold" generally between 10,000 and 1,000,000
        """
        self.alpha = alpha
        self.threshold = threshold

    def find_corners(self, image, *, border=20, n_corners=inf):
        """Apply the Harris detector to find corners in an image."""

        if isinstance(image, JpegImageFile):
            image = np.array(image)

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

        log_image(A, "out1")
        log_image(B, "out2")
        log_image(C, "out3")

        # "Make the Corner Response Function (CRF)"

        h, w = image.shape
        Q = np.zeros((h, w))

        for v in range(h):
            for u in range(w):

                a = A[v, u]
                b = B[v, u]
                c = C[v, u]

                det = (a * b) - (c * c)
                trace = a + b
                Q[v, u] = det - self.alpha * (trace * trace)

        log_image(Q, "out4")

        return A, B

    def __repr__(self):
        _attributes = ["alpha", "threshold"]
        return (
            self.__class__.__name__
            + "("
            + ", ".join([attr + "=" + str(getattr(self, attr)) for attr in _attributes])
            + ")"
        )
