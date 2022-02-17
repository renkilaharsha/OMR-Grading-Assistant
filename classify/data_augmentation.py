# data_augmentation.py: Tools for image data augmentation
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

Methods for performing data augmentation, assuming that handwritten
digits are invariant under small affine transformations.

Implementation Notes
--------------------

Alexander previously wrote something similar to the noise model when
estimating emission probabilities for a Monte Carlo simulation for
"Reading Handwritten Text."

This was implemented for David Crandall's Artificial Intelligence
course (https://github.iu.edu/cs-b551-sp2021/jkumari-jerpark-hayesall-a3).
"""

from random import randint
from random import choice
from random import uniform
from numpy.random import default_rng
import numpy as np
from PIL import Image


class NoiseFactory:
    """Create new examples by randomly sampling."""

    def __init__(self):
        self.noise_model = NoiseModel()

    def make_image(self, image):

        image = Image.fromarray(image)

        x = randint(-5, 5)
        y = randint(-5, 5)
        theta = randint(-15, 15)
        oper = choice(("xor", "or"))
        noise = uniform(0.01, 0.1)

        transformed = np.array(affine_transform(image, x, y, theta)) > 128
        noisy = self.noise_model.add_noise(transformed, operation=oper, noise=noise)
        return Image.fromarray((noisy * 255).astype(np.uint8))


def affine_transform(image, x=0, y=0, rot=0):
    """
    Return a new image translated by ``(x, y)``
    and rotated by ``rot`` degrees.

    Arguments:
        image (PIL.Image): The image to transform
        x: The x-coordinate of the translation (default: 0)
        y: The y-coordinate of the translation (default: 0)
        rot: The rotation in degrees (default: 0)

    Returns:
        A copy of the image with transformations applied
    """
    return image.transform(
        image.size,
        Image.AFFINE,
        (1, 0, x, 0, 1, y),
        Image.BICUBIC,
        fillcolor=(255,),
    ).rotate(rot, Image.BICUBIC, fillcolor=(255,))


class NoiseModel:
    """
    The NoiseModel assumes boolean images.
    """

    def __init__(self, noise=0.05):
        """Initialize the NoiseModel.

        Arguments:
            noise: Binomial distribution parameter (default: 0.05)
        """
        self.noise = noise
        self._rng = default_rng()

    def add_noise(self, image, operation="xor", noise=None):
        """
        Return a copy of the image with noise applied.

        Arguments:
            image (np.ndarray): A bool array
        """
        if noise is None:
            amount = self.noise

        _noise = self._rng.binomial(1, noise, size=image.shape)

        if operation == "xor":
            return np.bitwise_xor(image, _noise)
        elif operation == "or":
            return np.bitwise_or(image, _noise)

        raise ValueError(f"Unknown operation: {operation}")
