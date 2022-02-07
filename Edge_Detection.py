from PIL import Image
import math
from multiprocessing import Pool
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Edgedetection:
    def __init__(self,image_path,lowtheshold=50,highthreshold=100):
        self.image = Image.open(image_path).convert('L') #convert a gray scale
        self.width, self.height = self.image.size
        print(self.height,self.width)
        self.image_numpy = np.asarray(self.image)
        self.low_threshold = lowtheshold
        self.high_threshold  =highthreshold
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



    def return_gradient_direction(self,gx,gy):
        """
        This function computes the gradient direction for the non_maximum supression
        :param gx: x direction gradient
        :param gy: y direction gradient
        :return: the coordinates of the neighbouring pixels along gradient
        """
        if(gx==0):
            return 0,+1,0,-1
        else:
            theta  = np.rad2deg(np.arctan(gy/gx))
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

    def check_potential_edge(self,conv_x,conv_y,i,j):
        """
        :param conv_x: sobel operator horizantal kernal out put on image
        :param conv_y: sobel operator vertical kernal out put on image
        :param i: x index
        :param j: y index
        :return: returns whether the edge is potential or not
        """
        a,b,c,d = self.return_gradient_direction(conv_x[i,j],conv_y[i,j])
        gm = (abs(conv_x[i,j])**2 + abs(conv_y[i,j])**2)**0.5
        gm1= (abs(conv_x[i+a,j+b]**2)+ abs(conv_y[i+a,j+b])**2)**0.5
        gm2 = (abs(conv_x[i+c,j+d])**2 + abs(conv_y[i+c,j+d])**2)**0.5
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
        nms_ht = nms_array.copy() # non_maximum_supression high threshold array
        nms_lt = nms_array.copy() # non_maximum_supression high threshold array
        nms_ht[nms_ht>self.high_threshold]=1
        nms_lt[nms_lt<self.low_threshold]=0
        for i in range(pad,len(nms_array)-(kernal_size-1)):
            for j in range(pad,len(nms_array[0])-(kernal_size-1)):
                if(nms_lt[i][j]!=0) :
                    if(nms_ht[i-1,j-1] ==1 or nms_ht[i+1,j+1] ==1 or nms_ht[i,j-1] ==1 or nms_ht[i,j+1] ==1 or nms_ht[i-1,j] ==1 or nms_ht[i+1,j] ==1 or nms_ht[i+1,j-1] ==1 or nms_ht[i-1,j+1] ==1):
                        nms_ht[i][j] = 1
                    else:
                        nms_ht[i][j] =0

        edge_link = np.multiply(self.image_numpy,nms_ht)
        return edge_link

    def image_smoothing(self,array,pad,kernal_size,sigma):
        smooth_image  =  np.zeros(array.shape)
        gauss_kernal = self.gaussian_kernal(kernal_size,sigma)
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
        for i in range(pad,len(array)-(kernal_size-1)):
            for j in range(pad,len(array[0])-(kernal_size-1)):
                conv_sub_matrix = array[i:i+kernal_size,j:j+kernal_size]
                gx = np.sum(np.multiply(kernal_horizantal,conv_sub_matrix))
                gy = np.sum(np.multiply(kernal_vertical,conv_sub_matrix))
                conv_array_x[i,j] = gx
                conv_array_y[i,j] = gy
        return conv_array_x,conv_array_y


    def non_max_supression(self,kernal_size=3,gauss_sigma=0.5):
        pad = math.floor(kernal_size/2)
        array = self.image_numpy
        smooth_image = self.image_smoothing(array,pad,kernal_size,gauss_sigma)
        im = Image.fromarray(smooth_image)
        plt.imsave("output/{}_gaussian_smoothing.png".format(self.image_name),im)
        conv_array_x,conv_array_y = self.sobel_convolution(smooth_image, kernal_size,pad, self.horizantal_sobel, self.vertical_sobel)
        edge_image = np.zeros(array.shape)
        conv_array = abs(conv_array_y)+ abs(conv_array_x)
        #conv_array[conv_array < 0] = 0
        #conv_array[conv_array > 255] = 255
        im = Image.fromarray(conv_array)
        plt.imsave("output/{}_sobel_edge_detection.png".format(self.image_name),im)
        for i in range(pad,len(array)-(kernal_size-1)):
            for j in range(pad,len(array[0])-(kernal_size-1)):
                edge_image[i,j] = self.check_potential_edge(conv_array_x,conv_array_y,i,j)
        #hough_transform(conv_array)
        im = Image.fromarray(edge_image)
        plt.imsave("output/{}_non_maximum_supression.png".format(self.image_name),im)
        edge_image = self.edge_linking(edge_image,pad,kernal_size)
        im = Image.fromarray(edge_image)
        plt.imsave("output/{}_final_edge_detection.png".format(self.image_name), im)
        return edge_image


    def hough_transform(self,img_matrix):
        h, w = img_matrix.shape
        d = int((h ** 2 + w ** 2) ** 0.5)
        height_half, width_half = int(h / 2), int(w / 2)
        hough_transform_coordinates = np.zeros((d, 180))
        theta_range = np.arange(180)
        cos_theta = np.cos(np.deg2rad(theta_range))
        sin_theta = np.sin(np.deg2rad(theta_range))
        for i in range(len(img_matrix)):
            for j in range(len(img_matrix[0])):
                if (img_matrix[i][j] !=0):
                    for theta in range(len(theta_range)):
                        row = int((j - height_half) * (cos_theta[theta]) + (i - width_half) * (sin_theta[theta]))
                        hough_transform_coordinates[row][theta_range[theta]] += 1



if __name__ == '__main__':
    #image_path = "test-images/a-30.jpg"
    image_path = "example.jpg"
    detection = Edgedetection(image_path)
    edge_image = detection.non_max_supression()
#edge_image = np.multiply(edge_image,I)
#edge_image[edge_image==1]=255


#plt.imshow(im3)

