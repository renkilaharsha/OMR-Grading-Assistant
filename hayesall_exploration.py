"""
Order of operations:

1. Find horizontal lines
2. Find vertical lines
3. Find intersections (these should give us a strictly increasing set of coordinates)
4. Use the intersections to find corners/"bounding boxes"
5. Use the bounds in a downstream process for identifying
    A/B/C/D/E, or combinations thereof.
"""


# %% Setup packages for exploration.

from PIL import Image
from PIL import ImageFilter
import numpy as np
from matplotlib.pyplot import imshow
from statistics import mode
from heapq import heappush, heappop

# %% Define a KNN method and Data class:

class KNN:
    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X):
        heap = []
        for i in range(self.X.shape[0]):
            heappush(heap, (np.linalg.norm(self.X[i] - X), self.y[i]))
        print(heap)
        return mode([heappop(heap)[1] for _ in range(self.k)])

def unflatten(flat):
    return flat.reshape((44, 277))

# %% Let's load some "training data"
# Load the a-27 test image, crop out particular regions of interest.
# We'll keep some extra variables around for now for ease of visualization.

im = Image.open("test-images/a-27.jpg")

cropped_a_1 = im.crop((250, 1153, 527, 1197))
cropped_a_2 = im.crop((250, 1389, 527, 1433))

cropped_b_1 = im.crop((250, 773, 527, 817))
cropped_b_2 = im.crop((250, 821, 527, 865))

cropped_c_1 = im.crop((250, 869, 527, 913))
cropped_c_2 = im.crop((250, 1201, 527, 1245))

cropped_d_1 = im.crop((250, 678, 527, 722))
cropped_d_2 = im.crop((250, 726, 527, 770))

cropped_e_1 = im.crop((685, 1481, 962, 1525))
cropped_e_2 = im.crop((1117, 1005, 1394, 1049))

a1 = np.array(cropped_a_1)
a2 = np.array(cropped_a_2)
b1 = np.array(cropped_b_1)
b2 = np.array(cropped_b_2)
c1 = np.array(cropped_c_1)
c2 = np.array(cropped_c_2)
d1 = np.array(cropped_d_1)
d2 = np.array(cropped_d_2)
e1 = np.array(cropped_e_1)
e2 = np.array(cropped_e_2)

data = [a1, a2, b1, b2, c1, c2, d1, d2, e1, e2]

X_train = np.array([arr.flatten() for arr in data])
y_train = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])

# %% Do some learning maybe.

clf = KNN(k=1)

clf.fit(X_train, y_train)

print("Predicting on training example:")
clf.predict(X_train[0])

# %% Let's see what happens when we fetch the second image.

im2 = Image.open("test-images/a-3.jpg")

cropped_example = im2.crop((263, 676, 540, 720))
test_example = np.array(cropped_example).flatten()

X_test = np.array([test_example, test_example])

print("Predicting on test example:")
print(clf.predict(X_test[0]))

# %%
edge_image = im.filter(ImageFilter.FIND_EDGES)

edge_image.save("edge_image.jpg")
