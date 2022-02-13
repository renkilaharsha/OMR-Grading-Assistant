# Copyright 2022 Alexander L. Hayes
# Computer Vision - CSCI-B 657 - Spring 2022
# MIT License

from harris.harris import HarrisCornerDetector
from PIL import Image
import numpy as np

im = Image.open("docs/corner_demo_2.png")
im2 = np.array(im)

hcd = HarrisCornerDetector(alpha=0.04)

corners = hcd.find_corners(im2)
