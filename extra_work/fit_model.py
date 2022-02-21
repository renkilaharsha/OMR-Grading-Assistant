# learn_model.py: Learn a Naive Bayes classifier with data augmentation
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
Script for training a Bayesian Network model for recognizing whether an
image contains a handwritten letter.

- This creates a ``model.pkl`` file for later use.

- Some information about the model is printed to the console.

- This shows *three examples* to the user, as well as the *model's prediction*.

    - A "positive" example, which is a handwritten letter.
    - A "negative" example, which is a randomly-sampled patch of `test-images/a-3.jpg`.
      (a-3 contained *no* handwritten digits, so it was safe to sample patches from it)
    - An "out of distribution" example, which contains the letters "ABC" and was
      not represented at all in the training data.
      (This might give us some insight into whether it's generalizing well).
"""

import os
from random import shuffle

from classify.data_augmentation import NoiseFactory
from classify.naive_bayes import NaiveBayesClassifier

from PIL import Image
import numpy as np
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt


def load_data():
    """Load the training data, leave one pos/neg example out for testing."""

    pos_files = os.listdir("classify/training_data/positives")
    shuffle(pos_files)

    neg_files = os.listdir("classify/training_data/negatives")
    shuffle(neg_files)

    print("\tFound {} positive examples and {} negative examples.\n".format(len(pos_files), len(neg_files)))

    pos_files_train = pos_files[:-1]
    neg_files_train = neg_files[:-1]

    X = []
    y = []

    fact = NoiseFactory()

    for file in pos_files_train:
        im = Image.open("classify/training_data/positives/" + file)

        X.append(np.array(im).flatten())
        y.append(1)

        for _ in range(150):
            im_array = np.array(im)
            generated = np.array(fact.make_image(im_array))

            X.append(generated.flatten())
            y.append(1)

    for file in neg_files_train:
        im = Image.open("classify/training_data/negatives/" + file)

        X.append(np.array(im).flatten())
        y.append(0)

        for _ in range(150):
            im_array = np.array(im)
            generated = np.array(fact.make_image(im_array))

            X.append(generated.flatten())
            y.append(0)

    X = (np.array(X) > 128).astype(np.float64)
    y = np.array(y)

    print("\tTraining data shape: {}\n".format(X.shape))

    return X, y, pos_files[-1], neg_files[-1]


if __name__ == "__main__":

    print("Loading data. Make sure `classify/training_data` exists, or unzip it.\n")
    print("\tThis generates 150 'augmented' images per positive and negative example.\n")
    X, y, test_pos, test_neg = load_data()

    print("Training a model.\n")
    clf = NaiveBayesClassifier()

    clf.fit(X, y)

    print("Saving the model to `model.pkl`.\n")
    clf.save("model.pkl")

    final_pos_image = Image.open("classify/training_data/positives/" + test_pos)
    final_neg_image = Image.open("classify/training_data/negatives/" + test_neg)
    weird_image = Image.open("../docs/handwritten/weird_case_abc.png")

    print("I'll demonstrate performance on three unseen examples:\n")

    print("(True label: 1) Test pos, predicted:   ", clf.predict((np.array(final_pos_image) > 128).flatten().reshape(1, -1)))
    print("(            0) Test neg, predicted:   ", clf.predict((np.array(final_neg_image) > 128).flatten().reshape(1, -1)))
    print("(            1) Unseen case, predicted:", clf.predict((np.array(weird_image) > 128).flatten().reshape(1, -1)))

    imshow(final_pos_image)
    plt.show()

    imshow(final_neg_image)
    plt.show()

    imshow(weird_image)
    plt.show()
