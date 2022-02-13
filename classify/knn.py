# Copyright 2022 Alexander L. Hayes
# Computer Vision - CSCI-B 657 - Spring 2022
# MIT License

"""
Simple KNN classifier
"""

from statistics import mode
from heapq import heappush, heappop
import numpy as np


class KNNClassifier:
    """
    Simple KNN classifier.
    """

    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X):
        heap = []
        for i in range(self.X.shape[0]):
            heappush(heap, (np.linalg.norm(self.X[i] - X), self.y[i]))
        return mode([heappop(heap)[1] for _ in range(self.k)])
