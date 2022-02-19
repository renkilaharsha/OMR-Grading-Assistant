# Copyright 2022 Alexander L. Hayes
# Computer Vision - CSCI-B 657 - Spring 2022
# MIT License

"""
A set of hand-crafted rules that decides whether a snippet from an image is
A/B/C/D/E or combinations of those.
"""

from .naive_bayes import NaiveBayesClassifier
from .utils import make_crop, split_crop, split_right, onehot_to_label
import numpy as np


def silly_classify(image, x, y):

    _crop = make_crop(image, x, y)

    _, _, right = split_crop(_crop)

    a, b, c, d, e = split_right(right)

    return onehot_to_label([np.sum(k) < 500_000 for k in [a, b, c, d, e]])


class RuleBasedIdentifier:
    """A rule-based system for producing '44 ABC x' from an image."""

    def __init__(self, model_path="classify/model.pkl"):
        try:
            self.model = NaiveBayesClassifier.load(model_path)
        except FileNotFoundError:
            raise RuntimeError(
                "Could not instantiate `RuleBasedIdentifier`."
                "\n\n\t`classify/model.pkl` not found.\n\n"
                "\tEither run `fit_model.py` to create a `model.pkl`,\n"
                "\tor check the `model_path=` parameter in "
                "`RuleBasedIdentifier`:\n\n"
                "\t```\n"
                "\tRuleBasedIdentifier(model_path='classify/model.pkl')\n"
                "\t```"
            )

    def identify(self, number, image):
        """Identify the letters and whether a handwritten letter is in the left."""

        # TODO(hayesall): Tuneable parameters:
        #   - Threshold for the bubbles

        if not isinstance(image, np.ndarray):
            image = np.array(image)

        left, _, right = split_crop(image)

        handwritten = self.model.predict(
            (left > 128).flatten().reshape(1, -1)
        )

        a, b, c, d, e = split_right(right)
        bubbles = onehot_to_label([np.sum(k) < 500_000 for k in [a, b, c, d, e]])

        if handwritten[0]:
            return f"{number} {bubbles} x"
        return f"{number} {bubbles}"
