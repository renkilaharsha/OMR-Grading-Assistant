from PIL import Image
import math
from multiprocessing import Pool
import numpy as np
from matplotlib.pyplot import cm
import pandas as pd
import matplotlib.pyplot as plt

class Edgedetection:
    def __init__(self,image_path,lowthesholdratio=0.7,highthresholdratio=0.8):
        self.image = Image.open(image_path).convert('L') #convert a gray scale
        self.width, self.height = self.image.size
        print(self.height,self.width)
        self.image_numpy = np.asarray(self.image)
        self.low_threshold = lowthesholdratio
        self.high_threshold  =highthresholdratio
        self.vertical_sobel  = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        self.horizantal_sobel = np.array([[-1, 0, 1], [-2, 0, 1], [-1, 0, 1]])
        self.image_name  =  image_path.split(".")[0].split("/")[-1]

    def gaussian_kernal(self,kernal_size,sigma,is_log=False):
        """
        Taken reference from the below link
        https://en.wikipedia.org/wiki/Canny_edge_detector#Non-maximum_suppression
        https://stackoverflow.com/questions/29731726/how-to-calculate-a-gaussian-kernel-matrix-efficiently-in-numpy
        :param kernal_size:
        :return:
        """
        if(is_log==True):
            return(np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])) #laplacian of gaussian filter
        ### taken from above stackoverflow link
        values =  np.linspace(-(kernal_size-1)/2, (kernal_size-1)/2, num=kernal_size)
        gauss = np.exp(-0.5 * np.square(values) / np.square(sigma)).reshape(1,kernal_size)
        kernal = np.matmul(gauss.T,gauss)
        return (kernal/np.sum(kernal))


    def normalize(self,img):
        img  = img/np.max(img)
        return img

    def angle_normalizer(self,grad_direction):
        arr = grad_direction.copy()
        for i in range(len(arr)):
            for j in range(len(arr[0])):
                theta = arr[i,j]
                if (theta > 22.5 and theta <= 67.5):
                    arr[i,j] = 45
                if (theta > 67.5 and theta <= 112.5):
                    arr[i, j] = 90
                if (theta > 112.5 and theta <= 157.5):
                    arr[i, j] = 45
                if (theta > 157.5 and theta <= 202.5):
                    arr[i, j] = 0
                if (theta > 337.5 or theta <= 22.5):
                    arr[i, j] = 0
        return arr

    def return_gradient_direction(self,theta):
        """
        This function computes the gradient direction for the non_maximum supression
        :param gx: x direction gradient
        :param gy: y direction gradient
        :return: the coordinates of the neighbouring pixels along gradient
        """

        if(theta > 22.5 and theta<=67.5):
            return -1 , +1 ,1,-1
        if (theta > 67.5 and theta <= 112.5):
            return -1, 0, 1, 0
        if (theta > 112.5 and theta <= 157.5):
            return -1, -1, 1, 1
        if (theta > 157.5 and theta <= 202.5):
            return 0, -1, 0, 1
        if (theta > 202.5 and theta <= 247.5):
            return 1, -1, -1, 1
        if (theta > 247.5 and theta <= 292.5):
            return 1, 0, -1, 0
        if (theta > 292.5 and theta <= 337.5):
            return 1, 1, -1, -1
        if (theta > 337.5 or theta <= 22.5):
            return 0, +1, 0, -1

    def check_potential_edge(self,grad_magnitude,grad_direction,i,j):
        """
        :param conv_x: sobel operator horizantal kernal out put on image
        :param conv_y: sobel operator vertical kernal out put on image
        :param i: x index
        :param j: y index
        :return: returns whether the edge is potential or not
        """
        a,b,c,d = self.return_gradient_direction(grad_direction[i,j])
        gm = grad_magnitude[i,j]
        gm1= grad_magnitude[i+a,j+b]
        gm2 = grad_magnitude[i+c,j+d]
        if(  gm1 > gm or gm2 > gm ):
            return 0
        else:
            return gm

    def edge_linking(self,nms_array,pad,kernal_size):
        """
        This function performs the hysterisis/ edgelinking afetr rrrnon maximum supression
        :param nms_array: non maximun supressed array
        :param pad:
        :param kernal_size:
        :return:
        """
        self.high_threshold = np.max(nms_array)*self.high_threshold
        self.low_threshold = np.max(nms_array)*self.low_threshold
        nms_ht = nms_array.copy() # non_maximum_supression high threshold array
        nms_lt = nms_array.copy() # non_maximum_supression high threshold array
        nms_ht[nms_ht>=self.high_threshold]=1
        nms_lt[nms_lt<self.low_threshold]=0
        for i in range(pad,len(nms_array)-(kernal_size-1)):
            for j in range(pad,len(nms_array[0])-(kernal_size-1)):
                if(nms_lt[i][j]!=0) :
                    if(nms_ht[i-1,j-1] ==1 or nms_ht[i+1,j+1] ==1 or nms_ht[i,j-1] ==1 or nms_ht[i,j+1] ==1 or nms_ht[i-1,j] ==1 or nms_ht[i+1,j] ==1 or nms_ht[i+1,j-1] ==1 or nms_ht[i-1,j+1] ==1):
                        nms_ht[i][j] = 1
                    else:
                        nms_ht[i][j] =0

        edge_link = nms_ht*255
        #edge_link = np.multiply(self.image_numpy,nms_ht)
        return edge_link

    def image_smoothing(self,array,pad,kernal_size,sigma,is_log=False):
        smooth_image  =  np.zeros(array.shape)
        gauss_kernal = self.gaussian_kernal(kernal_size,sigma,is_log)
        for i in range(pad,len(array)-(kernal_size-1)):
            for j in range(pad,len(array[0])-(kernal_size-1)):
                image_sub_matrix = array[i:i+kernal_size,j:j+kernal_size]
                smooth_image[i,j] = np.sum(np.multiply(gauss_kernal,image_sub_matrix))
        return(smooth_image)

    def sobel_convolution(self,array,kernal_size,pad,kernal_horizantal,kernal_vertical):
        """
        Taken reference fron the below link
        https://towardsdatascience.com/tensorflow-for-computer-vision-how-to-implement-convolutions-from-scratch-in-python-609158c24f82
        :param array:
        :param kernal_size:
        :param pad:
        :param kernal_horizantal:
        :param kernal_vertical:
        :return:
        """
        conv_array_x = np.zeros(array.shape)
        conv_array_y = np.zeros(array.shape)
        grad_direction = np.zeros(array.shape)
        for i in range(pad,len(array)-(kernal_size-1)):
            for j in range(pad,len(array[0])-(kernal_size-1)):
                conv_sub_matrix = array[i:i+kernal_size,j:j+kernal_size]
                conv_array_x[i,j] = np.sum(np.multiply(kernal_horizantal,conv_sub_matrix))
                conv_array_y[i,j] = np.sum(np.multiply(kernal_vertical,conv_sub_matrix))
                if( conv_array_x[i,j] != 0 ):
                    degrees =  np.degrees(np.arctan(conv_array_y[i,j]/conv_array_x[i,j]))
                    grad_direction[i,j] =  degrees if degrees >= 0 else 90-degrees
                    print(grad_direction[i,j])
        grad_magnitude = (conv_array_y**2 + conv_array_x**2)**0.5
        return grad_magnitude,grad_direction


    def edge_detection(self,kernal_size=3,gauss_sigma=0.5):
        pad = math.floor(kernal_size/2)
        array = 255 - self.image_numpy
        smooth_image = self.image_smoothing(array,pad,kernal_size,gauss_sigma,is_log=False)
        im = Image.fromarray(smooth_image)
        plt.imsave("output/{}_gaussian_smoothing.png".format(self.image_name),im,cmap=cm.gray)
        gradient_magnitude_xy, gradient_direction_xy = self.sobel_convolution(smooth_image, kernal_size,pad, self.horizantal_sobel, self.vertical_sobel)
        edge_image = np.zeros(array.shape)
        im = Image.fromarray(gradient_magnitude_xy)
        plt.imsave("output/{}_sobel_edge_detection.png".format(self.image_name),im,cmap=cm.gray)
        for i in range(pad,len(array)-(kernal_size-1)):
            for j in range(pad,len(array[0])-(kernal_size-1)):
                edge_image[i,j] = self.check_potential_edge(gradient_magnitude_xy,gradient_direction_xy,i,j)
        #hough_transform(conv_array)
        im = Image.fromarray(edge_image)
        plt.imsave("output/{}_non_maximum_supression.png".format(self.image_name),im,cmap=cm.gray)
        edge_image = self.edge_linking(edge_image,pad,kernal_size)
        '''max_pixel_value = np.max(edge_image)
        min_pixel_value = np.min(edge_image)
        edge_image = (edge_image - min_pixel_value / max_pixel_value - min_pixel_value) * 255'''
        #edge_image = 255-edge_image
        im = Image.fromarray(edge_image)
        plt.imsave("output/{}_final_edge_detection.jpg".format(self.image_name), im,cmap=cm.gray)
        grad_normalize = self.angle_normalizer(gradient_direction_xy)
        return edge_image, gradient_direction_xy

    def hough_transform_voting(self,img_matrix,gradient_direction):
        h, w = img_matrix.shape
        d_max = int((h ** 2 + w ** 2) ** 0.5)
        d_values = np.linspace(-d_max,d_max,d_max+d_max+1)
        theta_values = np.linspace(-90,90,181)
        hough_transform_coordinates = np.zeros((len(d_values),len(theta_values) ))
        for i in range(len(img_matrix)):
            for j in range(len(img_matrix[0])):
                if (img_matrix[i][j] !=0):
                    print(gradient_direction[i][j])
                    row = i*np.cos(np.deg2rad(gradient_direction[i][j])) + j*np.sin(np.deg2rad(gradient_direction[i][j]))
                    d = round(row)
                    itheta = round(gradient_direction[i][j])
                    if(row<0):
                        d = round(d_max - row)
                    #print(itheta,row)
                    hough_transform_coordinates[d][itheta]+=1
        return hough_transform_coordinates




if __name__ == '__main__':
    image_path = "test-images/c-18.jpg"
    #image_path = "example.jpg"
    #image_path = "canny.png"
    detection = Edgedetection(image_path)
    edge_image,gradient_xy = detection.edge_detection(gauss_sigma=5)
    voting = detection.hough_transform_voting(edge_image,gradient_xy)
    df = pd.DataFrame(voting.astype(int))
    print(df.head())
    df.to_csv("voting_c-18.csv",header=False,index=False)
#edge_image = np.multiply(edge_image,I)
#edge_image[edge_image==1]=255


#plt.imshow(im3)

