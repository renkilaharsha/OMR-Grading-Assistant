# Computer Vision - CSCI-B 657 - Spring 2022

"""
Inject secret solutions into an image. 😈😈😈
"""

import argparse
from PIL import Image
from PIL import ImageOps
import numpy as np


def get_answer_file_content(file_location):
    with open(file_location, "r") as fh:
        data_raw = fh.read().splitlines()
    return [line.strip("x").strip(" ").split() for line in data_raw]


def encode_solution_vector(input_data):

    dict_options = {'A':0,'B':1,'C':2,'D':3,'E':4}

    list_encoded = []

    for i in input_data:
        list_chars= [char for char in i[1]]
        list_encoded.append([dict_options[char] for char in list_chars])

    # Convert the vector into a numpy array
    solution_vector = np.zeros((5, 85))
    for idx1, chars in enumerate(list_encoded):
        for char in chars:
            solution_vector[char, idx1] = 200

    return solution_vector


def forward(vec):
    """Roll vectors in two dimensions to obfuscate the solutions."""
    return np.roll(np.roll(vec, 1, 0), 2, 1)


if __name__ == "__main__":

    PARSER = argparse.ArgumentParser(description='Inject the solutions into an image.')
    PARSER.add_argument("image", help="The image to inject the solutions into.")
    PARSER.add_argument("solutions", help="The solutions to inject.")
    PARSER.add_argument("output", type=str, default="injected.jpg", help="Output image file.")
    ARGS = PARSER.parse_args()

    list_all_ans = get_answer_file_content(ARGS.solutions)

    vec = encode_solution_vector(list_all_ans)

    vec_encode = forward(vec)

    im = np.array(ImageOps.grayscale(Image.open(ARGS.image)))

    # Insert an expanded version of array into the form.
    im[400:400+vec.shape[0] * 4, 400:400 + vec.shape[1] * 4] = np.repeat(np.repeat(np.repeat(np.repeat(vec_encode, 2, axis=0), 2, axis=1), 2, axis=0), 2, axis=1)

    # Let's insert an additional pixel at 400 - 4, 400 - 4
    im[392:396, 392:396] = 0
    im[396:400, 396:400] = 0

    # Convert the numpy array back into a form and we'll save it.
    Image.fromarray(im).save(ARGS.output)
