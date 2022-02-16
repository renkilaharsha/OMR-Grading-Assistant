# Classifying Snippets of an Image

This module implements some tools for converting `(44, 400)` crops of an image
into a human-readable A/B/C/D/E representation.

## Minimal Example:

```python
from classify.rule_based import silly_classify
from PIL import Image

im = Image.open("test-images/b-27.jpg")

print(silly_classify(im, 582, 1333))
# ABC
```

## Assumptions

This assumes you can pass a *reasonable* top-left coordinate the image.

## Short Notes

- `classify.utils`: Utilities for cropping images and converting 5-collections into "ABCDE" strings
- `classify.knn`: An unoptimized nearest neighbor implementation
- `classify.rule_based`: Simple rule-based methods for classifying regions.
