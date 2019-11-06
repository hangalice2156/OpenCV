import sys

import cv2
import numpy as np
import math
import random

from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.uic import loadUi

import tensorflow as tf

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

def gaussian_filter(img):
    x, y = np.mgrid[-1:2, -1:2]
    sigma = 1
    gaussian_kernel = np.exp(-(x**2+y**2)/(2*sigma**2))
    # normalize
    gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()

    img_filtered = np.copy(img)
    # iterating img's x and y axis
    for i in range(1,img.shape[0]-1):
        for j in range(1,img.shape[1]-1):
            temp = 0
            # applying gaussian filter
            for k in range(0,3):
                for m in range(0,3):
                    temp  = temp + (gaussian_kernel[k][m] * img[i+k-1][j+m-1])
            img_filtered[i][j] = temp

    return img_filtered

def sobel_edge_detection(img, kernel):
    img_sobel = np.copy(img)
    pdf = np.zeros(256)
    cdf = np.zeros(256)
    map = np.zeros(256)

    # applying histogram
    # probability density function
    for i in range(0,img_sobel.shape[0]):
        for j in range(0,img_sobel.shape[1]):
            pdf[img_sobel[i][j]] = pdf[img_sobel[i][j]] + 1

    # cumulative distribution function
    cdf[0] = pdf[0]
    for i in range(1,256):
        cdf[i] = pdf[i] + cdf[i-1]

    # mapping
    for i in range(0,256):
        map[i] = round((cdf[i]-min(cdf)) / ((img_sobel.shape[0] * img_sobel.shape[1]) - min(cdf)) * 255)
    for i in range(0,img_sobel.shape[0]):
            for j in range(0,img_sobel.shape[1]):
                img_sobel[i][j] = map[img_sobel[i][j]]

    padded_image = np.zeros((img.shape[0] + (2 * 1), img.shape[1] + (2 * 1)))
    padded_image[1:padded_image.shape[0] - 1, 1:padded_image.shape[1] - 1] = img

    # applying edge detection
    for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                img_sobel[i][j] = np.sum(kernel * padded_image[i:i+3, j:j+3])
    
    return img_sobel

class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        loadUi('./F74056328_hw1.ui', self)

        # section 1
        self.pushButton_LI.clicked.connect(self.load_image)
        self.pushButton_CC.clicked.connect(self.color_conversion)
        self.pushButton_IF.clicked.connect(self.image_flipping)
        self.pushButton_BL.clicked.connect(self.blending)
        # section 2
        self.pushButton_GT.clicked.connect(self.global_threshold)
        self.pushButton_LT.clicked.connect(self.local_threshold)
        # section 3
        self.pushButton_RST.clicked.connect(self.transforms)
        self.pushButton_PT.clicked.connect(self.perspective_transformation)
        # section 4
        self.pushButton_GA.clicked.connect(self.gaussian)
        self.pushButton_SOX.clicked.connect(self.sobel_x)
        self.pushButton_SOY.clicked.connect(self.sobel_y)
        self.pushButton_MAG.clicked.connect(self.magnitude)
        # section 5
        self.pushButton_STI.clicked.connect(self.show_train_images)

    def load_image(self):
        img = cv2.imread('./images/dog.bmp')
        cv2.imshow('dog', img)
        print('Height: ' + str(img.shape[0]))
        print('Width: ' + str(img.shape[1]))

    def color_conversion(self):
        img = cv2.imread('./images/color.png')
        img_cc = np.zeros_like(img)
        img_cc[..., 0] = img[..., 1] # change red into green
        img_cc[..., 1] = img[..., 2] # g into blue
        img_cc[..., 2] = img[..., 0] # blue into green
        cv2.imshow("color", img)
        cv2.imshow("color conversion", img_cc)

    def image_flipping(self):
        img = cv2.imread('./images/dog.bmp')
        img_flip_hz = cv2.flip(img, 1) # horizontal
        img_flip_vt = cv2.flip(img, 0) # vertical
        img_flip_di = cv2.flip(img, -1) # both
        cv2.imshow('dog', img)
        cv2.imshow('god', img_flip_hz)
        cv2.imshow('qod', img_flip_vt)
        cv2.imshow('bop', img_flip_di)

    def blending(self):
        img = cv2.imread('./images/dog.bmp')
        img_flip = cv2.flip(img, 1)
        
        def trackbar(value):
            img_blend = cv2.addWeighted(img, value/100, img_flip, 1-value/100, 0.0)
            cv2.imshow('blending', img_blend)
        
        cv2.imshow('blending', img)
        cv2.createTrackbar('blend','blending', 0, 100, trackbar)

    def global_threshold(self):
        img = cv2.imread('./images/QR.png')
        cv2.imshow('QRcode', img)

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert image into gray scale
        ret, img_result = cv2.threshold(img_gray, 80, 255, cv2.THRESH_BINARY)
        if ret:
            cv2.imshow('global', img_result)
        else:
            print('image load failed while applying global threshold.')

    def local_threshold(self):
        img = cv2.imread('./images/QR.png')
        cv2.imshow('QRcode', img)

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert image into gray scale
        img_result_mean = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 19, -1)
        img_result_gu = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 19, -1)
        cv2.imshow('mean', img_result_mean)
        cv2.imshow('gaussian', img_result_gu)

    def transforms(self):
        angle = float(self.angle_edit.text()) if self.angle_edit.text() else 0
        scale = float(self.scale_edit.text()) if self.scale_edit.text() else 1
        tx = float(self.tx_edit.text()) if self.tx_edit.text() else 0
        ty = float(self.ty_edit.text()) if self.ty_edit.text() else 0

        img = cv2.imread('./images/OriginalTransform.png')
        transform_array = cv2.getRotationMatrix2D((tx, ty), angle, scale)
        img_transform = cv2.warpAffine(img, transform_array, img.shape[:2])

        cv2.imshow('original', img)
        cv2.imshow('transformed', img_transform)

    def perspective_transformation(self):
        img = cv2.imread('./images/OriginalPerspective.png')
        cv2.imshow('OP', img)
        clicked_points = []
        target_point = [(20, 20), (20, 450), (450, 450), (450, 20)]
        
        def mouse_click_left(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and len(clicked_points) < 4:
                clicked_points.append((x, y))
            
            if len(clicked_points) == 4:
                transform_array = cv2.getPerspectiveTransform(np.float32(clicked_points), np.float32(target_point))
                img_transform = cv2.warpPerspective(img, transform_array,(430,430))
                cv2.imshow('Perspective Transformation', img_transform)
                clicked_points.clear()
                
        cv2.setMouseCallback('OP', mouse_click_left)

    def gaussian(self):
        img = cv2.imread('./images/School.jpg')
        cv2.imshow('original', img)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imshow('gray', img_gray)
        img_gaussian = gaussian_filter(img_gray)

        cv2.imshow('gaussian', img_gaussian)

    def sobel_x(self):
        img = cv2.imread('./images/School.jpg')
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_gaussian = gaussian_filter(img_gray)
        cv2.imshow('gaussian', img_gaussian)

        gx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        img_edge_x = sobel_edge_detection(img_gaussian, gx)

        cv2.imshow('sobel_x', img_edge_x)

    def sobel_y(self):
        img = cv2.imread('./images/School.jpg')
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_gaussian = gaussian_filter(img_gray)
        cv2.imshow('gaussian', img_gaussian)

        gy = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
        img_edge_y = sobel_edge_detection(img_gaussian, gy)

        cv2.imshow('sobel_y', img_edge_y)

    def magnitude(self):
        img = cv2.imread('./images/School.jpg')
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_gaussian = gaussian_filter(img_gray)

        gx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        gy = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
        img_edge_x = sobel_edge_detection(img_gaussian, gx)
        img_edge_y = sobel_edge_detection(img_gaussian, gy)
        img_magnitude = np.copy(img)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                img_magnitude[i][j] = math.sqrt(int(img_edge_x[i][j])*int(img_edge_x[i][j]) + int(img_edge_y[i][j])*int(img_edge_y[i][j]))

        cv2.imshow('magnitude', img_magnitude)

    def show_train_images(self):
        for i in range(0, 10):
            image_num = random.randint(0, len(x_train))
            first_train_img = np.reshape(x_train[image_num, :], (28, 28))
            plt.matshow(first_train_img, cmap = plt.get_cmap('gray'))
            plt.show()
            print(y_train[image_num])

if __name__ == '__main__':
    main()