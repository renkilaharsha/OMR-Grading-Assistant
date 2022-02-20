import sys
from edge_detection.detection import OMRDetection



if __name__ == '__main__':
    omr = OMRDetection(image_path=sys.argv[1])
    edge_image, gradient_xy = omr.edge_detection(gauss_sigma=5)
    voting = omr.hough_transform_voting(edge_image, gradient_xy)
    omr.inject_answers(sys.argv[2],voting,sys.argv[3])
