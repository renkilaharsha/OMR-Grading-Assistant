# Classifying Snippets of an Image

This module implements some tools for converting `(44, 400)` crops of an image
into a human-readable A/B/C/D/E representation.

## Finding A/B/C/D/E Answers:

```python
from classify.rule_based import silly_classify
from PIL import Image

im = Image.open("test-images/b-27.jpg")

print(silly_classify(im, 582, 1333))
# ABC
```

## Detecting Handwritten Letters

`classify.utils.split_crop` left portion is a `(70, 44)` region,
or a `(3080,)` feature vector.

A Naive Bayes model for detecting whether this region contains handwritten
letters (1) or not (0).

Here's a complete example for this image:

![Handwritten ABC example](../docs/handwritten/weird_case_abc.png)

```python
from classify.naive_bayes import NaiveBayesClassifier
import numpy as np
from PIL import Image

clf = NaiveBayesClassifier.load("classify/model.pkl")

# Create a training example
im = Image.open("docs/handwritten/weird_case_abc.png")

# The classifier operates on binarized (0, 1) images. `reshape` is necessary since we only have one example
vec = (np.array(im) > 128).flatten().reshape(1, -1)

print(clf.predict(vec))
# [1]
```

The usage is similar for arbitrary numpy arrays sampled from a bigger image.

## Assumptions

This assumes you can pass a *reasonable* top-left coordinate the image.

## Short Notes

- `classify.utils`: Utilities for cropping images and converting 5-collections into "ABCDE" strings
- `classify.knn`: An unoptimized nearest neighbor implementation
- `classify.rule_based`: Simple rule-based methods for classifying regions.
