# template_matching.py: Methods for finding templates in a bigger image.
# Computer Vision - CSCI-B 657 - Spring 2022

# Copyright © 2022 Alexander L. Hayes (hayesall)
# Copyright © 2019, the scikit-image team
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:

# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in
#    the documentation and/or other materials provided with the
#    distribution.
# 3. Neither the name of skimage nor the names of its contributors may be
#    used to endorse or promote products derived from this software without
#    specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
# IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
# IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""
Overview
--------

Methods for performing "template matching," or finding things that
are similar to a "template image" (R) in a "search image" (I).

See: "Principles of Digital Image Processing: Core Algorithms,"
Chapter 11, Comparing Images, pp. 258-264.

Implementation Notes
--------------------

I implemented the ``match_correlation_coefficient`` function.
This should be equivalent to the correlation coefficient definition
from Chapter 11, p. 262; however, this is so painfully slow that that
I was strongly considering replacing it with a Rust implementation.

When I peaked into whether template matching existed in off-the-shelf libraries,
scikit-image had one that was significantly faster, but uses a
fast fourier transform convolution and some clever vectorization
tricks that I'm less familiar with.

The ``match_template`` function below is almost identical to the one
implemented in ``scikit-image``. I've included the copyright
statement above,
"""

import math
import numpy as np
from scipy.signal import fftconvolve


def match_correlation_coefficient(image, template):
    """
    Compute the correlation coefficient between an image and a template.

    Arguments:
        image (np.ndarray): The image to search.
        template (np.ndarray): The template to search for.

    Returns:
        Matrix of correlation coefficients.
    """
    hi, wi = image.shape
    hr, wr = template.shape

    # Set the image and templates to be between 0 and 1.
    image = image / 255.0
    template = template / 255.0

    K = wr * hr
    sum_r = 0.0
    sum_r2 = 0.0

    for i in range(hr - 1):
        for j in range(wr - 1):
            sum_r += template[i, j]
            sum_r2 += template[i, j] ** 2

    r_bar = sum_r / K
    sr = np.sqrt(sum_r2 - ((sum_r2 ** 2) / K))

    C = np.zeros((hi - hr + 1, wi - wr + 1))

    for r in range(hi - hr):
        for s in range(wi - wr):

            # Compute the correlation coefficient for position (r, s)
            sum_i = 0.0
            sum_i2 = 0.0
            sum_ir = 0.0

            for i in range(hr - 1):
                for j in range(wr - 1):
                    ai = image[r + i, s + j]
                    ar = template[i, j]
                    sum_i += ai
                    sum_i2 += ai ** 2
                    sum_ir += ai * ar

            C[r, s] = (sum_ir - (sum_i * r_bar)) / (sr * np.sqrt(sum_i2 - ((sum_i2 ** 2) / K)))

    return C



def match_template(image, template):
    """Match a template to an image using normalized correlation.
    The output is an array with values between -1.0 and 1.0. The value at a
    given position corresponds to the correlation coefficient between the image
    and the template.

    Parameters
    ----------

    image : (M, N[, D]) array
        Input image.
    template : (m, n[, d]) array
        Template to locate. It must be `(m <= M, n <= N[, d <= D])`.

    Returns
    -------
    output : array
        Response image with correlation coefficients.

    Notes
    -----

    Details on the cross-correlation are presented in [1]_. This implementation
    uses FFT convolutions of the image and the template. Reference [2]_
    presents similar derivations but the approximation presented in this
    reference is not used in our implementation.

    References
    ----------
    .. [1] J. P. Lewis, "Fast Normalized Cross-Correlation", Industrial Light
           and Magic.
    .. [2] Briechle and Hanebeck, "Template Matching using Fast Normalized
           Cross Correlation", Proceedings of the SPIE (2001).
           :DOI:`10.1117/12.421129`
    """

    # Academic Integrity Statement: notes below this were based on the
    # `scikit-image` implementation of template matching. The version
    # implemented above works but it significantly slower.
    # Alexander has included additional discussion about this in the
    # "Implementation Notes" section above.

    def _window_sum_2d(image, window_shape):

        window_sum = np.cumsum(image, axis=0)
        window_sum = (window_sum[window_shape[0]:-1] - window_sum[:-window_shape[0] - 1])

        window_sum = np.cumsum(window_sum, axis=1)
        window_sum = (window_sum[:, window_shape[1]:-1] - window_sum[:, :-window_shape[1] - 1])

        return window_sum

    if image.ndim < template.ndim:
        raise ValueError("Dimensionality of template must be less than or "
                         "equal to the dimensionality of image.")
    if np.any(np.less(image.shape, template.shape)):
        raise ValueError("Image must be larger than template.")

    image_shape = image.shape

    image = image.astype(np.float64, copy=False)

    pad_width = tuple((width, width) for width in template.shape)
    image = np.pad(image, pad_width=pad_width, mode='constant', constant_values=0)

    image_window_sum = _window_sum_2d(image, template.shape)
    image_window_sum2 = _window_sum_2d(image ** 2, template.shape)

    template_mean = template.mean()
    template_volume = math.prod(template.shape)
    template_ssd = np.sum((template - template_mean) ** 2)

    xcorr = fftconvolve(image, template[::-1, ::-1], mode="valid")[1:-1, 1:-1]

    numerator = xcorr - image_window_sum * template_mean

    denominator = image_window_sum2
    np.multiply(image_window_sum, image_window_sum, out=image_window_sum)
    np.divide(image_window_sum, template_volume, out=image_window_sum)
    denominator -= image_window_sum
    denominator *= template_ssd
    np.maximum(denominator, 0, out=denominator)
    np.sqrt(denominator, out=denominator)

    response = np.zeros_like(xcorr, dtype=np.float64)

    mask = denominator > np.finfo(np.float64).eps

    response[mask] = numerator[mask] / denominator[mask]

    slices = []
    for i in range(template.ndim):
        d0 = template.shape[i] - 1
        d1 = d0 + image_shape[i] - template.shape[i] + 1
        slices.append(slice(d0, d1))

    return response[tuple(slices)]
