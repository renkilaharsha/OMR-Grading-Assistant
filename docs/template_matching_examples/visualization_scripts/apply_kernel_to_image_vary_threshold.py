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
from template_matching.template_matching import match_template
from template_matching.kernels import make_k1, make_k2, make_k3, blur_kernel
from template_matching.filters import blur_filter

from classify.utils import split_crop
from classify.utils import make_crop
from classify.rule_based import RuleBasedIdentifier

import matplotlib.pyplot as plt
from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage
from matplotlib import rc


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
                output[u-2:u+3, v-2:v+3] = 1

    return output



def nan_helper(y):
    return np.isnan(y), lambda z: z.nonzero()[0]


def oned_signal(output):

    output_sums = np.zeros(output.shape[1])
    for i in range(output.shape[1]):
        output_sums[i] = np.sum(output[:, i])

    output_sums = convolve1d(output_sums, blur_filter, axis=0)

    return output_sums

def interpolate_answers(output, hoffset=0, voffset=0):
    # Estimate which column of has the most points in it by summing sequences of columns:
    output_sums = np.zeros(output.shape[1])
    for i in range(1, output.shape[1] - 1):
        output_sums[i] = np.sum(output[:, i-1:i+2])

    # Blur the sums to get rid of noise.
    output_sums = convolve1d(output_sums, blur_filter, axis=0)

    # Find the peaks in the smoothed sums.
    peaks, _ = find_peaks(output_sums, height=3)

    # Set all columns to zero if they are not in the peaks.
    # i.e. 0 : 1st , 1st : 2nd, ..., n-1 : n
    peaks = [0] + list(peaks) + [output_sums.shape[0]]

    for p1, p2 in zip(peaks, peaks[1:]):
        output[:, p1+5:p2-5] = 0

    final_coordinates = None
    n_expected = [7, 7, 7]
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


rbi = RuleBasedIdentifier()

im_raw = Image.open("corrupted_example.png")
im = np.array(im_raw)

A = convolve1d(convolve1d(im, blur_filter, axis=1), blur_filter, axis=0)


def save_image(data, filename):
    sizes = np.shape(data)
    fig = plt.figure()
    fig.set_size_inches(1. * sizes[1] / sizes[0], 1, forward = False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(data)
    plt.savefig(filename, dpi = sizes[0], cmap='hot')
    plt.close()

# save_image(output, "OUTPUT.png")

# coordinates = interpolate_answers(output)

########################
## Visualization Code ##
########################

rc('font', **{'family': 'monospace'})

kernel = blur_kernel(make_k1())
out = match_template(A, kernel)

fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(8, 4))
plt.tight_layout()

# Total = DURATION × FPS
DURATION = 10
FPS = 10

def make_frame(t):

    threshold = t / 10

    ax0.clear()
    ax1.clear()

    ax1.set_title("K1 Kernel. Threshold: {:.2f}".format(threshold))

    output = collect_local_maxima(out, threshold=threshold)
    onedsignal = oned_signal(output)

    ax0.imshow(output, cmap='inferno')
    ax1.plot(onedsignal)

    ax0.axis('off')
    ax1.axis('off')

    return mplfig_to_npimage(fig)

animation = VideoClip(make_frame, duration=DURATION)
animation.write_gif('k1_kernel_signal.gif', fps=FPS)
