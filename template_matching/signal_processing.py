# signal_processing.py: 1D interpolation for finding peaks
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

This signal processing techniques to find peaks in a template-matched
matrix, and can help with interpolating the peaks for where peaks
are *likely* to occur.
"""

from .filters import blur_filter

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


def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Academic Integrity Statement:
        This was pulled from a StackOverflow post for interpolating
        numpy arrays with NaNs.
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
