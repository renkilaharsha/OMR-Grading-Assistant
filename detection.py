from PIL import Image
import math
import numpy as np
from matplotlib.pyplot import cm
import pandas as pd
import matplotlib.pyplot as plt

class OMRDetection:
    def __init__(self,image_path,lowthesholdratio=0.7,highthresholdratio=0.9):
        self.image = Image.open(image_path).convert('L') #convert a gray scale
        self.width, self.height = self.image.size
        print(self.height,self.width)
        self.image_numpy = np.asarray(self.image)
        self.low_threshold = lowthesholdratio
        self.high_threshold  =highthresholdratio
        self.vertical_sobel  = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        self.horizantal_sobel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        self.image_name  =  image_path.split(".")[0].split("/")[-1]
        self.lable_dict =  {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"}


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
        if(  gm > gm1 and gm > gm2 ):
            return gm
        else:
            return 0

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
                if(nms_ht[i,j]!=1):
                    if(nms_lt[i][j]!=0) :
                        if(nms_ht[i-1,j-1] ==1 or nms_ht[i+1,j+1] ==1 or nms_ht[i,j-1] ==1 or nms_ht[i,j+1] ==1 or nms_ht[i-1,j] ==1 or nms_ht[i+1,j] ==1 or nms_ht[i+1,j-1] ==1 or nms_ht[i-1,j+1] ==1):
                            nms_lt[i][j] = 1
                        else:
                            nms_lt[i][j]=0
                else:
                    nms_lt[i,j]=1

        edge_link = nms_lt
        self.sub_image = self.image_substarction(edge_link,255-self.image_numpy)
        #edge_link = np.multiply(self.image_numpy,nms_ht)
        return edge_link

    def image_substarction(self,edge_image,image):
        sub_image = np.zeros((edge_image.shape))
        for i in range(len(sub_image)):
            for j in range(len(sub_image[0])):
                if(sub_image[i,j]!=1):
                    sub_image[i,j] = image[i,j]
        return sub_image


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
        grad_magnitude = (conv_array_y**2 + conv_array_x**2)**0.5
        return grad_magnitude,grad_direction


    def edge_detection(self,kernal_size=3,gauss_sigma=0.5):
        pad = math.floor(kernal_size/2)
        array = 255-self.image_numpy
        smooth_image = self.image_smoothing(array,pad,kernal_size,gauss_sigma,is_log=False)
        print("Smoothing of image completed")
        im = Image.fromarray(smooth_image)
        plt.imsave("output/{}_gaussian_smoothing.png".format(self.image_name),im,cmap=cm.gray)
        gradient_magnitude_xy, gradient_direction_xy = self.sobel_convolution(smooth_image, kernal_size,pad, self.horizantal_sobel, self.vertical_sobel)
        print("Sobel convolution on image completed")
        edge_image = np.zeros(array.shape)
        im = Image.fromarray(gradient_magnitude_xy)
        plt.imsave("output/{}_sobel_edge_detection.png".format(self.image_name),im,cmap=cm.gray)
        for i in range(pad,len(array)-(kernal_size-1)):
            for j in range(pad,len(array[0])-(kernal_size-1)):
                edge_image[i,j] = self.check_potential_edge(gradient_magnitude_xy,gradient_direction_xy,i,j)
        print("Non max supression of edges completed")
        #hough_transform(conv_array)

        im = Image.fromarray(edge_image)
        plt.imsave("output/{}_non_maximum_supression.png".format(self.image_name),im,cmap=cm.gray)
        edge_image = self.edge_linking(edge_image,pad,kernal_size)
        im = Image.fromarray(edge_image)
        plt.imsave("output/{}_final_edge_detection.jpg".format(self.image_name), im,cmap=cm.gray)
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
                    row = i*np.cos(np.deg2rad(gradient_direction[i][j])) + j*np.sin(np.deg2rad(gradient_direction[i][j]))
                    d = round(row)
                    itheta = round(gradient_direction[i][j])
                    if(row<0):
                        d = round(abs(row))
                    #print(itheta,row)
                    hough_transform_coordinates[d][itheta]+=1
        return hough_transform_coordinates

    def detect_vertical_horizantal_lines(self,lines_array,voting,angle):
        """
        This function find all lines horizantal/vertical from hough transfroms and give the bounding lines required.
        :param lines_array:
        :param voting:
        :param angle:
        :return:
        """
        i = 0
        while i < len(lines_array):
            start = i
            index = start
            flag = 1
            while flag:
                if (voting[index, angle] > 10):
                    index += 1
                else:
                    flag = 0
            if (index - start > 10):
                lines_array[start, angle] = 1
                lines_array[index, angle] = 1
            i += index - start + 1
            flag = 1

    def remove_extra_lines(self,horizantal):
        k = []
        for i in range(len(horizantal)):
            if (horizantal[i] < 500):
                k.append(i)

        for i in range(len(k)):
            horizantal.pop(k[0])

    def extract_coordinates(self,horizantal,vertical):
        verti = 0
        horiz = 0
        question_dict = {}
        questions = 1
        while questions < 30:
            if (questions < 28):
                for i in range(3):
                    question_dict[questions + i * 29] = [(vertical[verti], horizantal[horiz]),
                                                         (vertical[verti + 10 - 1], horizantal[horiz + 1])]
                    verti = verti + 10
            else:
                for i in range(2):
                    question_dict[questions + i * 29] = [(vertical[verti], horizantal[horiz]),
                                                         (vertical[verti + 10], horizantal[horiz + 1])]
                    verti = verti + 10

            verti = 0
            horiz = horiz + 2
            questions += 1

        for questions in question_dict:
            print(questions, " : ", question_dict[questions])
        return question_dict

    def extract_markings(self,voting,edge_image,filename):
        lines_array = np.zeros((voting.shape))
        self.detect_vertical_horizantal_lines(lines_array,voting,0)
        self.detect_vertical_horizantal_lines(lines_array,voting,90)
        horizantal = list(np.where(lines_array[:, 0] == 1)[0])
        vertical = list(np.where(lines_array[:, 90] == 1)[0])
        self.remove_extra_lines(horizantal)
        print("No of horizantal lines : ",len(horizantal), " , No of vertical lines : ", len(vertical))
        question_coordinates = self.extract_coordinates(horizantal,vertical)
        marking_dict = self.get_pixel_markings(edge_image, horizantal, vertical)
        keys = sorted(marking_dict.keys())

        file1 = open(filename, "w")
        for i in keys:
            file1.write("{} {}\n".format(str(i), marking_dict[i]))
        file1.close()
        return question_coordinates, marking_dict

    def count_pixels(self,v1, h1, v2, h2, image):
        sum = 0
        for i in range(v1, v2):
            for j in range(h1, h2):
                if (image[j, i] > 0):
                    sum += 1
        return sum

    def retrive_answers_file(self,file):
        """
        this reading line is taken from https://stackabuse.com/read-a-file-line-by-line-in-python/
        :param file:
        :return:
        """
        answers_dict = dict()
        file_pointer = open(file,mode="r")
        read_line = file_pointer.readline().strip()
        if (len(read_line) > 2):
            data = read_line.split()
            answers_dict[int(data[0])] = list(data[1].upper())
        while read_line:
            read_line = file_pointer.readline().strip()
            if (len(read_line) > 2):
                data = read_line.split()
                answers_dict[int(data[0])] = list(data[1].upper())
        file_pointer.close()
        return answers_dict


    def shade_answers(self,v1,h1,v2,h2,array):
        array[h1:h2,v1:v2] =0


    def inject_answers(self,answer_file,voting,injection_file):
        array =  self.image_numpy.copy()
        array.setflags(write=True)

        answers_dict = self.retrive_answers_file(answer_file)
        lines_array = np.zeros((voting.shape))
        self.detect_vertical_horizantal_lines(lines_array, voting, 0)
        self.detect_vertical_horizantal_lines(lines_array, voting, 90)
        horizantal = list(np.where(lines_array[:, 0] == 1)[0])
        vertical = list(np.where(lines_array[:, 90] == 1)[0])
        self.remove_extra_lines(horizantal)
        print("No of horizantal lines : ", len(horizantal), " , No of vertical lines : ", len(vertical))
        questions = 1
        verti = 0
        horiz = 0
        while questions < 30:
            if (questions < 28):
                for p in range(3):
                    for i in range(0, 10, 2):
                        if( self.lable_dict[int(i / 2)] in  answers_dict[questions + p * 29]):
                            self.shade_answers(vertical[verti + i] - 2, horizantal[horiz] - 2,
                                              vertical[verti + i + 1] + 5,
                                              horizantal[horiz + 1] + 5, array)
                    verti = verti + 10
            else:
                for p in range(2):
                    for i in range(0, 10, 2):
                        if (self.lable_dict[int(i / 2)] in answers_dict[questions + p * 29]):
                            self.shade_answers(vertical[verti + i] - 2, horizantal[horiz] - 2,
                                               vertical[verti + i + 1] + 5,
                                               horizantal[horiz + 1] + 5,array)

                    verti = verti + 10

            verti = 0
            horiz = horiz + 2
            questions += 1

        im = Image.fromarray(array)
        plt.imsave(injection_file, im, cmap=cm.gray)


    def get_box_marked_lables(self,pixeles_highlited):
        print(pixeles_highlited)
        sorted_index = np.argsort(pixeles_highlited)
        labels = []
        for i in range(len(sorted_index)):
            threshold = 50
            max = np.max(pixeles_highlited)

            if (max-pixeles_highlited[sorted_index[i]] < threshold):
                print(pixeles_highlited[sorted_index[i]])
                if (self.lable_dict[sorted_index[i]] not in labels):
                    labels.append(self.lable_dict[sorted_index[i]])
        label = "".join(sorted(labels))
        print("Label : " ,label)
        return label

    def get_pixel_markings(self,image, horizantal, vertical):
        marking_dict = dict()
        questions = 1
        verti = 0
        horiz = 0
        while questions < 30:
            if (questions < 28):
                for p in range(3):
                    pixels_higlighted = []
                    for i in range(0, 10, 2):
                        pixels_higlighted.append(
                            self.count_pixels(vertical[verti + i], horizantal[horiz], vertical[verti + i + 1],
                                         horizantal[horiz + 1] , self.sub_image))
                    print("question no : ",questions + p * 29)
                    marking_dict[questions + p * 29] = self.get_box_marked_lables(pixels_higlighted)
                    verti = verti + 10
            else:
                for p in range(2):
                    pixels_higlighted = []
                    for i in range(0, 10, 2):
                        pixels_higlighted.append(
                            self.count_pixels(vertical[verti + i], horizantal[horiz], vertical[verti + i + 1],
                                              horizantal[horiz + 1] + 5, self.sub_image))
                    print("question no : ",questions + p * 29)
                    marking_dict[questions + p * 29] = self.get_box_marked_lables(pixels_higlighted)
                    verti = verti + 10

            verti = 0
            horiz = horiz + 2
            questions += 1
        return marking_dict


if __name__ == '__main__':
    image_path = "test-images/c-18.jpg"
    #image_path = "example.jpg"
    #image_path = "canny.png"
    omr = OMRDetection(image_path)
    edge_image,gradient_xy = omr.edge_detection(gauss_sigma=5)
    voting = omr.hough_transform_voting(edge_image,gradient_xy)
    question_coordinates, marking_dict = omr.extract_markings(voting,edge_image)

#edge_image = np.multiply(edge_image,I)
#edge_image[edge_image==1]=255


#plt.imshow(im3)

