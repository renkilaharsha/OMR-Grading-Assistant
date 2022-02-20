from template_matching.template_matching import match_template
from template_matching.kernels import make_k3, blur_kernel
from template_matching.filters import blur_filter

from PIL import Image
import numpy as np
from scipy.ndimage import convolve1d
import matplotlib.pyplot as plt

def save_image(data, filename):
    sizes = np.shape(data)
    fig = plt.figure()
    fig.set_size_inches(1. * sizes[1] / sizes[0], 1, forward = False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(data, cmap='inferno')
    plt.savefig(filename, dpi = sizes[0])
    plt.close()

im_raw = Image.open("corrupted_example.png")
im = np.array(im_raw)

blurred_im = convolve1d(convolve1d(im, blur_filter, axis=1), blur_filter, axis=0)

save_image(blurred_im, "corrupted_example_blurred.png")

kernel = blur_kernel(make_k3())

matches = match_template(blurred_im, kernel)

save_image(matches, "corrupted_example_matches.png")
