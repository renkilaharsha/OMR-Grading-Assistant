# test_a1.py : Run A1 autograding tests, assuming 
#    grade.py, inject.py, extract.py exist per assignment instructions
# Author: Stephen Karukas

import subprocess
import os
from os.path import join
import inspect
from PIL import Image
import numpy as np
import re


LOG_DIR =  "autograding_log"
AUTOGRADING_FILES = "a1_autograding_files"
NUM_QUESTIONS = 85
MINUTE = 60


## utils

def run(test_name, time_limit_sec, argv):
    test_dir = join(LOG_DIR, test_name)
    os.makedirs(test_dir, exist_ok=True)

    stdout = open(join(test_dir, "stdout.txt"), "a+")
    stderr = open(join(test_dir, "stderr.txt"), "a+")
    argv = ["python3", *argv]
    msg = f"***** Running '{' '.join(argv)}' *****\n"
    stdout.write(msg)
    stderr.write(msg)
    stdout.flush()
    stderr.flush()

    try:
        process = subprocess.run(
            argv, 
            stderr=stderr, stdout=stdout, 
            timeout=time_limit_sec
        )
    except Exception as e:
        stdout.close()
        stderr.close()
        raise e
    return process.returncode


def read_detections(detection_file):
    # Example row: 1 ABCD x
    detections = [set()] * NUM_QUESTIONS

    for line in open(detection_file, "r"):
        line = line.strip()
        if line == "":
            continue

        numchars = ""
        i = 0
        while i < len(line) and line[i].isdigit():
            numchars += line[i]
            i += 1
        n = int(numchars)
        answer = line[i:]
        answer = set(c for c in "".join(answer).upper() if c in "ABCDEX")
        detections[n-1] = answer
    return detections


def count_matching_detections(ground_truth_file, detected_file):
    true_answers = read_detections(ground_truth_file)
    detected_answers = read_detections(detected_file)

    correct_detections = 0
    for gt, guess in zip(true_answers, detected_answers):
        if gt == guess:
            correct_detections += 1
        elif (gt - {'X'}) == (guess - {'X'}):
            # half credit for correct detections but missing the written-in letter
            correct_detections += 0.5

    return correct_detections
    

def grade_test(testname, im_fname):
    im_name = im_fname.split(".")[-2]
    im_fname = join(AUTOGRADING_FILES, im_fname)

    detection_fname = join(AUTOGRADING_FILES, im_name + "_detected.txt")
    ground_truth_fname = join(AUTOGRADING_FILES, im_name + "_groundtruth.txt")

    argv = ["grade.py", im_fname, detection_fname]
    run(testname, argv=argv, time_limit_sec=10*MINUTE)
    # compare detections
    return count_matching_detections(ground_truth_fname, detection_fname)


def transform_injected(fname, rotate_angle=0., p_noise=0., tx=0, ty=0):
    """
    Overwrite the image with a noisy/rotated version
    """
    if rotate_angle == p_noise == tx == ty == 0:
        return

    image = Image.open(fname)
    original_mode = image.mode

    image = image.convert("L")
    mat = tuple([
        1, 0, tx,
        0, 1, ty,
    ])
    image = image.transform(image.size, Image.AFFINE, mat, fillcolor=255)
    image = np.array(image.rotate(rotate_angle, fillcolor=255))
    image = image * (np.random.rand(*image.shape) > p_noise)

    # reconvert and save
    image = Image.fromarray(image).convert(original_mode)
    image.save(fname)



def inject_extract_test(testname, answers_fname, 
                        form_fname=join(AUTOGRADING_FILES, "blank_form.jpg"), 
                        rotate_angle=0., p_noise=0., tx=0, ty=0, filex=".png"):
    txt_name = answers_fname.split(".")[-2]

    answers_fname = join(AUTOGRADING_FILES, answers_fname)
    injected_fname = join(AUTOGRADING_FILES, txt_name + "_injected" + filex)
    extracted_fname = join(AUTOGRADING_FILES, txt_name + "_extracted.txt")
    
    # inject
    argv = ["inject.py", form_fname, answers_fname, injected_fname]
    run(testname, argv=argv, time_limit_sec=10*MINUTE)

    # simulate printing/scanning, replacing the injected image
    transform_injected(injected_fname, rotate_angle, p_noise, tx, ty)

    # extract
    argv = ["extract.py", injected_fname, extracted_fname]
    run(testname, argv=argv, time_limit_sec=10*MINUTE)

    # compare detections
    return count_matching_detections(answers_fname, extracted_fname)


## tests

def test_grade_1():
    test_name = inspect.stack()[0][3]
    return grade_test(test_name, "a-3.jpg")


def test_grade_2():
    test_name = inspect.stack()[0][3]
    return grade_test(test_name, "b-27.jpg")


def test_grade_3():
    test_name = inspect.stack()[0][3]
    return grade_test(test_name, "c-33.jpg")


def test_grade_4():
    test_name = inspect.stack()[0][3]
    return grade_test(test_name, "unseen-1.jpg")


def test_grade_5():
    test_name = inspect.stack()[0][3]
    return grade_test(test_name, "a-27.jpg")


def test_grade_6():
    test_name = inspect.stack()[0][3]
    return grade_test(test_name, "b-13.jpg")


def test_extract_1():
    test_name = inspect.stack()[0][3]
    return inject_extract_test(test_name, "a-3.txt")


def test_extract_2():
    test_name = inspect.stack()[0][3]
    return inject_extract_test(test_name, "a-48.txt", filex=".jpg")


def test_extract_3():
    test_name = inspect.stack()[0][3]
    return inject_extract_test(test_name, "c-18.txt")


def test_extract_4():
    test_name = inspect.stack()[0][3]
    return inject_extract_test(test_name, "b-27.txt")


def test_extract_noisy_1():
    test_name = inspect.stack()[0][3]
    return inject_extract_test(test_name, "b-13.txt", filex=".jpg", p_noise=0.001)


def test_extract_noisy_2():
    test_name = inspect.stack()[0][3]
    return inject_extract_test(test_name, "c-33.txt", p_noise=0.005, tx=4, ty=4)


def test_extract_noisy_3():
    test_name = inspect.stack()[0][3]
    return inject_extract_test(test_name, "a-30.txt", rotate_angle=0.5)


def test_extract_noisy_4():
    # 'badly scanned' test form
    my_name = inspect.stack()[0][3]
    form = join(AUTOGRADING_FILES, "a-30.jpg")
    return inject_extract_test(
        my_name, "a-30_2.txt", 
        filex=".jpg",
        form_fname=form,  p_noise=0.01, 
        rotate_angle=-1, tx=50, ty=50
    )



if __name__ == "__main__":
    # Run this file to run all the autograding tests
    tests = [fn_name for fn_name in dir() if fn_name.startswith("test_")]
    scores = {}

    for test_fn_name in tests:
        print(f"Running {test_fn_name}...")
        test_fn = globals()[test_fn_name]
        scores[test_fn_name] = test_fn()
        print(f"Got score {scores[test_fn_name]}")