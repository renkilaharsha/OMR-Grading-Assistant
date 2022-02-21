# Computer Vision - CSCI-B 657 - Spring 2022

"""
Extract secret solutions out of an image. 😈😈😈
"""

import argparse
from harris.harris import HarrisCornerDetector
from PIL import Image
from PIL import ImageOps
import numpy as np


def backward(vec):
    """Invert the `forward` function."""
    return np.roll(np.roll(vec, -1, 0), -2, 1)


def decode_solution_vector(vec):
    """Turn a numpy array back into answers."""
    dict_reverse = {0:'A',1:'B',2:'C',3:'D',4:'E'}
    list_final_ans = []
    for i in range(85):
        str_init = ''
        for j in range(5):
            if not vec[j,i]:
                str_init += dict_reverse[j]
        list_final_ans.append(str_init)

    return list_final_ans


if __name__ == "__main__":

    PARSER = argparse.ArgumentParser(description="Extract secret solutions out of an image.")
    PARSER.add_argument("image", type=str, default="injected.jpg", help="The image to extract the solutions from.")
    PARSER.add_argument("output", type=str, default="output.txt", help="Output to write text solutions to.")
    ARGS = PARSER.parse_args()

    raw_image = Image.open(ARGS.image)
    gray_image = ImageOps.grayscale(raw_image)
    inv_gray_image = ImageOps.invert(gray_image)

    # Let's find the top-left pixel in a region.
    hcd = HarrisCornerDetector()

    crop = inv_gray_image.crop((375, 375, 850, 450))
    np_crop = np.array(crop)

    out = np.ones(np_crop.shape) * 255

    corners = hcd.find_corners(np_crop)

    # We'll iterate over all possible corners and find the one closest to the top-left.
    all_corners = []
    for corner in corners:
        all_corners.append((corner, np.linalg.norm(np.array([0, 0] - corner.coords))))
    all_corners.sort(key=lambda x: x[1])
    top_left = all_corners[0][0]

    # We have the top-left pixel, we'll go 8-down, 8-right and extract the image.
    u, v = top_left.coords

    # Iterate over 85 columns and 5 rows.
    solution_vector = np.ones((5, 85))

    for i in range(85):
        for j in range(5):

            x = (u + 5) + (j * 4)
            y = (v + 5) + (i * 4)

            small = np_crop[x:x+4, y:y+4]
            solution_vector[j, i] = np.mean(small)

    solution_vector = backward(solution_vector > 100)

    list_final_ans = decode_solution_vector(solution_vector)

    with open(ARGS.output, "w") as fh:
        for i, ans in enumerate(list_final_ans):
            fh.write("{} {}\n".format(i+1, ans))
