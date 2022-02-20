import sys
import time

from edge_detection.detection import OMRDetection



if __name__ == '__main__':
    start_time  = time.time()
    omr = OMRDetection(image_path=sys.argv[1], output_file_path=sys.argv[2])
    omr.omr_extraction()
    print("Total time to extract : ", time.time()-start_time)