# Classifying Snippets of an Image

This module implements some tools for converting `(44, 400)` crops of an image
into a human-readable A/B/C/D/E representation.

## Finding "Bubbles" and "Handwritten Letters"

This combines the bubble step and the handwritten letter portion
into a common interface:

```python
from classify.rule_based import RuleBasedIdentifier
from classify.utils import make_crop
from PIL import Image

# Load an image:
im = Image.open("test-images/b-27.jpg")

# Instantiate the identifier:
identifier = RuleBasedIdentifier()

# Make a crop of the image with (582, 1333) as the top-left coordinate:
crop = make_crop(im, 582, 1333)

# Identify the bubbles for (44) and whether handwriting exists:
print(identifier.identify(44, crop))
```

**Output**:

`crop` contains a (44, 400) region:

![Local region showing a handwritten AB, 44, and ABC filled in](../docs/local_crop.png)

And the result of `identifier.identify` is:

```console
44 ABC x
```

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
