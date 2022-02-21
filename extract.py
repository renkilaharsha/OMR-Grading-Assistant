# Let's extract what we injected into the form.


from harris.harris import HarrisCornerDetector
from PIL import ImageOps
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import sys

injected_img = sys.argv[1]
output_file_name = sys.argv[2]
raw_image = Image.open(injected_img)
gray_image = ImageOps.grayscale(raw_image)
inv_gray_image = ImageOps.invert(gray_image)

# Let's find the top-left pixel in a region.
hcd = HarrisCornerDetector()

crop = inv_gray_image.crop((375, 375, 850, 450))
np_crop = np.array(crop)

# We'll empirically check every coordinate in the cropped image for the closest corner pixel.

out = np.ones(np_crop.shape) * 255

corners = hcd.find_corners(np_crop)

# We'll iterate over all possible corners and find the one closest to the top-left.
all_corners = []
for corner in corners:
    all_corners.append((corner, np.linalg.norm(np.array([0, 0] - corner.coords))))
all_corners.sort(key=lambda x: x[1])
top_left = all_corners[0][0]

# print(top_left)

plt.imshow(np_crop)

# We have the top-left pixel, we'll go 8-down, 8-right and extract the image.
u, v = top_left.coords

# Iterate over 85 columns and 5 rows.

solution_vector = np.ones((5, 85))

for i in range(85):
    for j in range(5):

        x = (u + 5) + (j * 4)
        y = (v + 5) + (i * 4)

        small = np_crop[x:x+4, y:y+4]
        solution_vector[j, i] = np.mean(small)

solution_vector = np.roll(solution_vector > 100, -17)


dict_reverse = {0:'A',1:'B',2:'C',3:'D',4:'E'}
list_final_ans = []
for i in range(85):
    str_init = ''
    for j in range(5):
        if not solution_vector[j,i]:
            str_init+=dict_reverse[j]
    list_final_ans.append(str_init)

# print(list_final_ans)
extract_text = open(output_file_name,'w')
for idx,ans in enumerate(list_final_ans):
    extract_text.write(f"{idx+1} {ans}\n")
extract_text.close()

