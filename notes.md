## Checking for the existence of handwritten digits

The assignment instructs:

> It should also output an x on a line for which it believes
> student has written in an answer to the left of the question (as the
> instructions on the answer form
> permit, but your program does not need to recognize which letter was
> written).

For example, if there appears to be a handwritten answer next to
question (6), then it should output an `x` on that line:

```text
1 A
2 A
3 B
4 B
5 C
6 AC x
```

### Making some observations

I went through the documents to see what sort of examples existed.

I observed A, B, C, E "singletons"

![Handwritten A](docs/handwritten/a-30_1015_1476_a.png)
![Handwritten B](docs/handwritten/a-30_1015_1335_b.png)
![Handwritten C](docs/handwritten/a-30_1015_1430_c.png)
![Handwritten E](docs/handwritten/b-13_555_1478_e.png)

I observed combinations of "AB" and "AE"

![Handwritten AB](docs/handwritten/b-13_987_767_ab.png)
![Handwritten AB](docs/handwritten/b-27_582_1333_ab.png)
![Handwritten AE](docs/handwritten/b-27_1010_1190_ae.png)

And I observed things I'll call "*false positives*"
in the `a-27` document. They're interesting because if we
naively assume that anything to the left of a problem is a signal to
regrade, we might incorrectly flag a lot of answers.

![False positive dot](docs/handwritten/a-27_130_1009_false.png)
![False positive dot](docs/handwritten/a-27_130_1104_false.png)
![False positive dot](docs/handwritten/a-27_130_1722_false.png)
![False positive dot](docs/handwritten/a-27_130_1910_false.png)
![False positive dot](docs/handwritten/a-27_562_1054_false.png)
![False positive dot](docs/handwritten/a-27_562_1242_false.png)

<details>
<summary>Coordinates and Documents (for reference)</summary>

```python
from classify.utils import make_crop
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
from PIL import Image

im = Image.open("test-images/b-27.jpg")

# Cases in b-27:
# imshow(make_crop(im, 582, 1333))      1
# imshow(make_crop(im, 1010, 1190))     1

# Cases in a-30:
# imshow(make_crop(im, 145, 1951))      1
# imshow(make_crop(im, 1015, 1619))     1
# imshow(make_crop(im, 1015, 1525))     1
# imshow(make_crop(im, 1015, 1476))     1
# imshow(make_crop(im, 1015, 1430))     1
# imshow(make_crop(im, 1015, 1382))     1

# Cases in a-27: (there will be false positives here if we're naive)
# imshow(make_crop(im, 130, 915))       0
# imshow(make_crop(im, 130, 962))       0
# imshow(make_crop(im, 130, 1009))      0
# imshow(make_crop(im, 130, 1152))      0
# imshow(make_crop(im, 130, 1722))      0
# imshow(make_crop(im, 562, 1054))      0
# imshow(make_crop(im, 562, 1242))      0
# imshow(make_crop(im, 562, 1575))      0
# imshow(make_crop(im, 1000, 1430))     0

# Cases in b-13:
# imshow(make_crop(im, 123, 912))       1
# imshow(make_crop(im, 123, 1290))      1
# imshow(make_crop(im, 555, 1478))      1

plt.show()
```

</details>

Therefore, we can make the following observations:

1. We're missing a handwritten "D"
2. We only have examples where "A" and another letter occur together. *Or*: we're missing a huge number of possible combinations
3. If we naively return "True" if there are pixels in a region, we'll produce false positives. *Or*: Not every mark is a letter.

### Building a Training Set

I'll assume we're interested in `(44, 70)` regions, or `(3080,)` flat
vectors.

```python
from classify.utils import make_crop, split_crop
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
from PIL import Image

im = Image.open("test-images/b-27.jpg")
left, _, _ = split_crop(make_crop(im, 1010, 1190))

im2 = Image.fromarray(left)

# Examples are still valid under small rotations:
im2.rotate(10, fillcolor=(255,))

# Examples are still valid under small affine transformations:
left = -5       # Small values (-5, 5)
up = 5          # Small values (-5, 5)
im2.transform(
    im2.size,
    Image.AFFINE,
    (1, 0, left, 0, 1, up),
    fillcolor=(255,),
)
```
