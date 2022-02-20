# Template Matching Methods

See the `template_matching.py` for implementation notes.

*Template Matching,* or finding things that are similar to a
"template image" (R) in a "search image" (I).

## Example Usage:

```python
from template_matching.template_matching import match_template
from template_matching.template_matching import match_correlation_coefficient
import numpy as np

# Search Image (I)
I = np.zeros((10, 10))
I[8, 8] = 1.0

# Template Image (R)
R = np.zeros((3, 3))
R[1, 1] = 1.0

print(match_template(I, R))
```

## References

- "Principles of Digital Image Processing: Core Algorithms," Chapter 11
