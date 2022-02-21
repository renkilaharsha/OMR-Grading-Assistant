from template_matching.template_matching import match_template
from template_matching.kernels import make_k3, blur_kernel
from template_matching.filters import blur_filter
from template_matching.signal_processing import (
    collect_local_maxima,
    interpolate_answers,
)

from classify.rule_based import RuleBasedIdentifier
from classify.utils import make_crop

from PIL import Image
from PIL import ImageOps
import numpy as np
from scipy.ndimage import convolve1d
import matplotlib.pyplot as plt


"""
import argparse

PARSER = argparse.ArgumentParser(description='Identify the answers in a scanned image.')
PARSER.add_argument("input_image", type=str, help="The input image to identify answers in.")
PARSER.add_argument("output_file", type=str, help="The output file to write the answers to.")
ARGS = PARSER.parse_args()
"""

class CorrelationCoefficientTemplateMatching:

    def __init__(self,filename,output_filename):
        self.rbi = RuleBasedIdentifier()
        self.raw_image = ImageOps.grayscale(Image.open(filename))
        self.outputfilename = output_filename

    def extract_markings(self):
        crop = self.raw_image.crop((0, 500, self.raw_image.width, self.raw_image.height))
        im = np.array(crop)

        # Blur the image to remove local noise.
        blurred_im = convolve1d(convolve1d(im, blur_filter, axis=1), blur_filter, axis=0)

        # Apply our template matching technique using our K3 kernel.
        kernel = blur_kernel(make_k3())
        matches = match_template(blurred_im, kernel)

        # Perform maximum suppression.
        local_maxima = collect_local_maxima(matches, threshold=0.7)

        # Interpolate the local maxima to find coordinates for where the
        # answers should be. This produces an (85, 2) vector or raises
        # an exception: suggesting the image violates assumptions.
        coordinates = interpolate_answers(local_maxima)

        # Apply our classifier on cropped regions near the coordinates.
        results = [
            self.rbi.identify(i + 1, make_crop(self.raw_image, x, y))
            for i, (x, y) in enumerate(coordinates)
        ]

        file1 = open(self.outputfilename, "w")

        for i in range(len(results)):
            file1.write("{}\n".format(results[i] ))
        file1.close()


'''for image in ["a-3", "a-27", "a-30", "a-48", "b-13", "b-27", "c-18", "c-33"]:

    im_raw = Image.open("../test-images/" + image + ".jpg")

    # We'll only look in the bottom portion of an image, since there
    # should not be answers above 500 pixels.
    crop = im_raw.crop((0, 500, im_raw.width, im_raw.height))
    im = np.array(crop)

    # Blur the image to remove local noise.
    blurred_im = convolve1d(convolve1d(im, blur_filter, axis=1), blur_filter, axis=0)

    # Apply our template matching technique using our K3 kernel.
    kernel = blur_kernel(make_k3())
    matches = match_template(blurred_im, kernel)

    # Perform maximum suppression.
    local_maxima = collect_local_maxima(matches, threshold=0.7)

    # Interpolate the local maxima to find coordinates for where the
    # answers should be. This produces an (85, 2) vector or raises
    # an exception: suggesting the image violates assumptions.
    coordinates = interpolate_answers(local_maxima)

    # Apply our classifier on cropped regions near the coordinates.
    results = [
        rbi.identify(i + 1, make_crop(im_raw, x, y))
        for i, (x, y) in enumerate(coordinates)
    ]

    # Estimate accuracy compared to ground truth file.
    with open(f"../test-images/{image}_groundtruth.txt", "r") as fh:
        ground_truth = fh.read().splitlines()

    results = np.array(results)
    ground_truth = np.array(ground_truth)
    print(f"{image}: {100 * np.sum(results == ground_truth) / len(ground_truth)}")'''
