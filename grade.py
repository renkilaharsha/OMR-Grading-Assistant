# !/usr/bin/env python

import argparse
import time
from edge_detection.detection import OMRDetection
from edge_detection.cctm import CorrelationCoefficientTemplateMatching


if __name__ == '__main__':
    start_time = time.time()
    PARSER = argparse.ArgumentParser(description='Identify the answers in a scanned image.')
    PARSER.add_argument("input_image", type=str, help="The input image to identify answers in.")
    PARSER.add_argument("output_file", type=str, help="The output file to write the answers to.")

    PARSER.add_argument('-m', '--method', nargs='?', const='specific method not given', help='Choose a method to extract the answers a) cctm b)edht')
    ARGS = PARSER.parse_args()

    if(ARGS.method== None):
        omr = CorrelationCoefficientTemplateMatching(ARGS.input_image,ARGS.output_file)
        omr.extract_markings()
    elif(ARGS.method == "edht"):
        omr = OMRDetection(image_path=ARGS.input_image, output_file_path=ARGS.output_file)
        omr.omr_extraction()

    print("Total time to extract : ", time.time() - start_time)