from classify.rule_based import silly_classify
from PIL import Image

im = Image.open("test-images/b-27.jpg")

print(silly_classify(im, 582, 1333))
