# naive_bayes.py: Learning and inference with Multinomial Naive Bayes classifiers.
# Copyright © 2021-2022 Alexander L. Hayes
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
Multiclass-class Naive Bayes Classifier

Overview
--------

Here is a flavor for how the syntax works (this should look familiar if you've
used ``scikit-learn`` before):

.. code-block:: python

    >>> from classify.naive_bayes import NaiveBayesClassifier
    >>> clf = NaiveBayesClassifier(alpha=1)
    >>> clf.fit(X_train, y_train)
    NaiveBayesClassifier(alpha=1)
    >>> clf.predict(X_test)
    [1 1 1 ... 0 0 0]

Academic Integrity Statement
----------------------------

Alexander L. Hayes originally wrote a two-class version of this for
Assignment 2 in David Crandall's "Artificial Intelligence" course in
Spring 2021, and submitted an N-class version that was almost
identical to portions of this for Assignment 3.
(https://github.iu.edu/cs-b551-sp2021/jkumari-jerpark-hayesall-a3)

Attribution
-----------

I was familiar with the Naive Bayes implementations from ``scikit-learn``.
This implements a N-class classification method similar to
``MultinomialNB``, and borrows some of the attribute and method names
from their documentation
(https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html).
I did not look at their code before/after implementing.

Gautam Kunapuli (one of Alexander's mentors) has a set of slides on Multinomial
Naive Bayes that Alexander looked at while implementing.
(https://gkunapuli.github.io/files/cs6375/09-NaiveBayes.pdf)

The "simple" example was pulled from Dan Jurafsky's slides on "Text
Classification and Naive Bayes." My first pass at implementing had a bug in it
(I missed taking the logarithm of the prior) and I debugged using the
conditional probabilities worked out on Slide 41.
(http://web.stanford.edu/~jurafsky/slp3/slides/7_NB.pdf#page=41)
"""

import pickle
import numpy as np


class NaiveBayesClassifier:
    """Multiclass NaiveBayesClassifier

    alpha :
        Laplace correction value (Default: 0.0)
    prior : None or array-like
        If None: fit the prior with data
    """

    def __init__(self, alpha=1.0, prior=None):
        self.alpha = alpha
        self.classes = None
        self.priors = prior
        self.probs = None

        if prior:
            assert np.sum(prior) == 1.0

    def save(self, filename):
        """Serialize self to a file."""
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(pklfile):
        """Return an instance of NaiveBayesClassifier from a saved pickle file."""
        with open(pklfile, "rb") as f:
            return pickle.load(f)

    def fit(self, X, y):
        # Warning: This assumes y is discretized into 0/1/2/...

        self.classes, _class_frequency = np.unique(y, return_counts=True)

        if not self.priors:
            self.priors = _class_frequency / np.sum(_class_frequency)

        _stacked = np.c_[y, X]

        self.probs = []
        for c in self.classes:
            _y = np.where(_stacked[:, 0] == c)
            prob = np.sum(_stacked[_y][:, 1:], axis=0)
            prob_denom = np.sum(prob) + prob.shape[0]
            self.probs.append(np.log((prob + self.alpha) / prob_denom))
        self.probs = np.array(self.probs)
        return self

    def predict(self, X):
        return np.argmax(np.log(self.priors) + np.dot(X, self.probs.T), axis=1)

    def __repr__(self):
        return (
            self.__class__.__name__
            + "(alpha="
            + str(self.alpha)
            + ")"
        )
