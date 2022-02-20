import sys
import time

from detection import OMRDetection



if __name__ == '__main__':
    start_time  = time.time()
    omr = OMRDetection(image_path=sys.argv[1])
    edge_image, gradient_xy = omr.edge_detection(gauss_sigma=5)
    voting = omr.hough_transform_voting(edge_image, gradient_xy)
    question_coordinates, marking_dict = omr.extract_markings(voting, edge_image,sys.argv[2])
    print("Total time to extract : ", time.time()-start_time)