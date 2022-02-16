# a1


I have implemented all the modules up to my knowledge.
All the modules are implemented from scratch using numpy.

Steps followed:
- Image smoothing using gaussian filter.
- Horizantal and Vertical edge detection using Sobel filter(3x3)
- Non-maximum supression.
- Hystereis/Edge linking. 
- Hough Transform line finding.
- Finding the intersection of lines using the lines extracted from above.
- finding the no of pixels in the each box.
- Filtering the box having non zero intensity pixel count greater than threshold.
- Transfering the filtered boxes to output format 


The program will take quite large amount of time like(3,4 minutes).
Need to optimize and refactor the code.

Grade operation is only implemented for shaded boxes not includes handwritten answers.(will update in next commit). As per i observed the grade operation is giving more than 90 % accuracy.

Usage :
```
python3 grade.py "test-images/b-27.jpg"  "output.txt"
```

```
python3 inject.py "test-images/blank_form.jpg" "test-images/a-3_groundtruth.txt" "injected_answers.jpg"
```
```
python3 extract.py "injected_answers.jpg"  "output.txt"
```

Any suggestions?........