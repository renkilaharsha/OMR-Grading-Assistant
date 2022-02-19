# Copyright 2022 Alexander L. Hayes
# Computer Vision - CSCI-B 657 - Spring 2022
# MIT License

"""
Short demo for working with the Harris Corner Detector.

- For `a-3`, threshold=10_000_000 seems reasonable
- For `book_corners1.png`, the default threshold is fine
"""

from harris.harris import HarrisCornerDetector
from PIL import Image
import numpy as np

im = np.array(Image.open("docs/book_corners1.png"))

hcd = HarrisCornerDetector(alpha=0.04, threshold=30000)

corners = hcd.find_corners(im)

print(corners)
