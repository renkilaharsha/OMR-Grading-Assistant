# Discussion with Alexander and Ajinkya

"""
Let's think through the injection portion first.
"""

from PIL import ImageOps
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import sys

blank_form = sys.argv[1]
answers_txt = sys.argv[2]
name_final_img = sys.argv[3]
f = open(answers_txt,'r')
list_all_ans = []

dict_options = {'A':0,'B':1,'C':2,'D':3,'E':4}

for i in range(85):
    content = f.readline()
    list_qs = content.strip().split(" ")
    list_all_ans.append(list_qs)


# print(list_all_ans)

list_encoded = []
for i in list_all_ans:
    list_chars= [char for char in i[1]]
    # print(list_chars)
    list_encoded.append([dict_options[char] for char in list_chars])

np_arr_mcq = np.ones((5, 85))

for idx1, chars in enumerate(list_encoded):
    for char in chars:
         np_arr_mcq[char, idx1] = 200


blank_form_img = Image.open(blank_form)
gray_image = ImageOps.grayscale(blank_form_img)
np_form = np.array(gray_image)

# Let's roll the array ahead by 17
np_array_mcq_rolled = np.roll(np_arr_mcq, 17)

# Insert an expanded version of array into the form.
np_form[400:400+np_arr_mcq.shape[0] * 4, 400:400 + np_arr_mcq.shape[1] * 4] = np.repeat(np.repeat(np.repeat(np.repeat(np_array_mcq_rolled, 2, axis=0), 2, axis=1), 2, axis=0), 2, axis=1)

# Let's insert an additional pixel at 400 - 4, 400 - 4
np_form[392:396, 392:396] = 0
np_form[396:400, 396:400] = 0

# We'll convert the numpy array back into a form and we'll save it.

Image.fromarray(np_form).save(name_final_img)
