import pdb
from template_matching.template_matching import match_template
from template_matching.kernels import make_k1, make_k2, make_k3, blur_kernel
from template_matching.filters import blur_filter

from classify.utils import split_crop
from classify.utils import make_crop
from classify.rule_based import RuleBasedIdentifier

import matplotlib.pyplot as plt


from PIL import Image

import numpy as np
from scipy.ndimage import convolve1d
from scipy.signal import find_peaks


def is_local_max(Q, u, v):
    return Q[u, v] == np.max(Q[u-3:u+5, v-3:v+5])


def collect_local_maxima(image, threshold=0.75, border=3):
    h, w = image.shape

    output = np.zeros((h, w))

    for u in range(border, h - border):
        for v in range(border, w - border):
            q = image[u, v]
            if q > threshold and is_local_max(image, u, v):
                output[u, v] = 1

    return output


kernel = blur_kernel(make_k3())


def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Source: https://stackoverflow.com/questions/6518811/interpolate-nan-values-in-a-numpy-array

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]


def interpolate_answers(output, hoffset=495, voffset=-110):
    """
    For an image where we've estimated the local maxima of the
    templates, find the 29 + 29 + 27 coordinates for the answers.

    Parameters
        output: The output of ``collect_local_maxima``

    Returns
        An ``(85, 2)`` array of coordinates where the answers are located.

    Assumptions:
        - 3 columns (peaks in a smoothed 1D signal)
        - The top and bottom coordinate of each column are "correct"
        - We're missing no more than two coordinates in a row
        - We might have too few coordinates, but NOT too many
    """

    # Estimate which column of has the most points in it by summing sequences of columns:
    output_sums = np.zeros(output.shape[1])
    for i in range(1, output.shape[1] - 1):
        output_sums[i] = np.sum(output[:, i-1:i+2])

    # Blur the sums to get rid of noise.
    output_sums = convolve1d(output_sums, blur_filter, axis=0)

    # Find the peaks in the smoothed sums.
    peaks, _ = find_peaks(output_sums, height=8)

    # Set all columns to zero if they are not in the peaks.
    # i.e. 0 : 1st , 1st : 2nd, ..., n-1 : n
    peaks = [0] + list(peaks) + [output_sums.shape[0]]

    for p1, p2 in zip(peaks, peaks[1:]):
        output[:, p1+5:p2-5] = 0

    final_coordinates = None
    n_expected = [29, 29, 27]
    for p, n_expect in zip(peaks[1:-1], n_expected):

        nx, ny = np.nonzero(output[:, p-5:p+5])
        ny += p

        x_points = [nx[0]]
        y_points = [ny[0]]

        if len(nx) > n_expect:
            raise RuntimeError("Too many points, this will fail.")

        # If there are less than 29 nx pairs, we need to interpolate.
        if len(nx) < n_expect:
            for i, (x1, x2) in enumerate(zip(nx, nx[1:])):

                if x2 - x1 > 105:
                    # Probably missing two points
                    x_points += [np.nan, np.nan]
                    y_points += [np.nan, np.nan]

                elif x2 - x1 > 50:
                    # Probably missing one point
                    x_points += [np.nan]
                    y_points += [np.nan]

                x_points.append(x2)
                y_points.append(ny[i+1])

            x_points = np.array(x_points)
            nans, x = nan_helper(x_points)
            x_points[nans] = np.interp(x(nans), x(~nans), x_points[~nans])

            y_points = np.array(y_points)
            nans, y = nan_helper(y_points)
            y_points[nans] = np.interp(y(nans), y(~nans), y_points[~nans])

            # We should have the same number of x and y coordinates.
            assert len(x_points) == len(y_points) == n_expect

            coordinates = np.array([y_points.astype(int), x_points.astype(int)]).T

        else:
            coordinates = np.array([ny.astype(int), nx.astype(int)]).T

        if final_coordinates is None:
            final_coordinates = coordinates
        else:
            final_coordinates = np.vstack((final_coordinates, coordinates))

    final_coordinates[:, 0] += voffset
    final_coordinates[:, 1] += hoffset

    return np.array(final_coordinates)

'''
import argparse

PARSER = argparse.ArgumentParser(description='Identify the answers in a scanned image.')
PARSER.add_argument("input_image", type=str, help="The input image to identify answers in.")
PARSER.add_argument("output_file", type=str, help="The output file to write the answers to.")
ARGS = PARSER.parse_args()
'''

rbi = RuleBasedIdentifier()

for image in ["a-3", "a-27", "a-30", "a-48", "b-13", "b-27", "c-18", "c-33"]:

    im_raw = Image.open("test-images/" + image + ".jpg")
    crop = im_raw.crop((0, 500, im_raw.width, im_raw.height))

    im = np.array(crop)
    A = convolve1d(convolve1d(im, blur_filter, axis=1), blur_filter, axis=0)

    out = match_template(A, kernel)
    output = collect_local_maxima(out, threshold=0.7)
    coordinates = interpolate_answers(output)

    results = []

    for i, (x, y) in enumerate(coordinates):

        local_crop = make_crop(im_raw, x, y)

        results.append(rbi.identify(i + 1, local_crop))

        # print(rbi.identify(i + 1, local_crop))

    with open(f"test-images/{image}_groundtruth.txt", "r") as fh:
        ground_truth = fh.read().splitlines()

    results = np.array(results)
    ground_truth = np.array(ground_truth)

    # Estimate accuracy compared to ground truth file.
    print(f"{image}: {100 * np.sum(results == ground_truth) / len(ground_truth)}")
