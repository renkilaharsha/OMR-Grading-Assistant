# Copyright 2022 Alexander L. Hayes
# Computer Vision - CSCI-B 657 - Spring 2022
# MIT License

"""
A set of hand-crafted rules that decides whether a snippet from an image is
A/B/C/D/E or combinations of those.
"""

from .utils import make_crop, split_crop, split_right, onehot_to_label
import numpy as np


def silly_classify(image, x, y):

    _crop = make_crop(image, x, y)

    _, _, right = split_crop(_crop)

    a, b, c, d, e = split_right(right)

    return onehot_to_label([np.sum(k) < 500_000 for k in [a, b, c, d, e]])


class HayesallRuleClassifier:

    def __init__(self):
        pass

    def fit(X, y):
        pass

    def predict(X):
        pass

