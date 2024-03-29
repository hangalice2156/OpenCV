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

import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

# dataset and parameters for LeNet
BATCH_SIZE = 256
LEARNING_RATE = 0.01
OPTIMIZER = 'Adam'
EPOCHS = 1
PRINT_FREQ = 100
TRAIN_NUMS = 49000

CUDA = True

if CUDA:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
print(device)

criterion = nn.CrossEntropyLoss()

kwargs = {}

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_t, y_t = torch.from_numpy(x_train.reshape(-1, 1, 28, 28)).float(), torch.from_numpy(y_train.astype('long'))
x_tt, y_tt = [
   torch.from_numpy(x_test.reshape(-1, 1, 28, 28)).float(),
   torch.from_numpy(y_test.astype('long'))
   ]

train_dataset = TensorDataset(x_t, y_t)
test_dataset = TensorDataset(x_tt, y_tt)

train_loader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=BATCH_SIZE, **kwargs)
test_loader = DataLoader(dataset=test_dataset, shuffle=True, batch_size=BATCH_SIZE, **kwargs)

loss_training = []
loss_validate = []
accuracy_validate = []
test_validate = None

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

def gaussian_filter(img):
    x, y = np.mgrid[-1:2, -1:2]
    sigma = 3
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

# Flatten function for CNN
def flatten(x):
    shape = x.size()
    x = x.view(shape[0], shape[1]*shape[2]*shape[3])
    return x

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
        self.pushButton_SH.clicked.connect(self.show_hyperparameters)
        self.pushButton_T1E.clicked.connect(self.train_one_epoch)
        self.pushButton_STR.clicked.connect(self.show_training_results)
        self.pushButton_INFER.clicked.connect(self.inference)

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
        plt.figure()
        for i in range(10):
            rand = random.randint(0, len(x_train))
            img = np.reshape(x_train[rand, :], (28, 28))
            plt.subplot(1, 10, i+1)
            plt.matshow(img, fignum=False, cmap=plt.get_cmap('gray'))
            plt.title(y_train[rand])
            plt.axis('off')
        plt.show()

    def show_hyperparameters(self):
        print('hyerparameters:')
        print('batch size: ' + str(BATCH_SIZE))
        print('learning rate: ' + str(LEARNING_RATE))
        print('optimizer: ' + OPTIMIZER)

    def train_one_epoch(self):
        global EPOCHS
        EPOCHS = 1
        trainer.train_loop(model, train_loader)
        trainer.test(model, test_loader)

        y_axis = np.arange(0, len(loss_training), 1)
        plt.plot(y_axis, loss_training[:])
        plt.xlabel('iteration')
        plt.ylabel('loss')
        plt.title('Train one EPOCH')
        plt.show()
        loss_training.clear()
        loss_validate.clear()
        accuracy_validate.clear()

    def show_training_results(self):
        img = cv2.imread('./Training_result.png')
        # if image not exist, train a new one then save show it
        if img is None:
            global EPOCHS
            EPOCHS = 50
            trainer.train_loop(model, train_loader)
            trainer.test(model, test_loader)

            plt.figure()

            plt.subplot(2, 1, 1)
            y_axis = np.arange(0, len(loss_validate), 1)
            plt.plot(y_axis, loss_validate[:])
            plt.xlabel('iteration')
            plt.ylabel('loss')
            plt.title('Train ' + str(EPOCHS) + ' EPOCH')

            plt.subplot(2, 1, 2)
            y_axis = np.arange(0, len(accuracy_validate), 1)
            plt.plot(y_axis, accuracy_validate[:])
            plt.xlabel('iteration')
            plt.ylabel('accuracy')

            plt.savefig('Training_result.png')
            plt.show()
            loss_training.clear()
            loss_validate.clear()
            accuracy_validate.clear()

            torch.save(model, './F74056166_model.pth')
        else:
            cv2.imshow('training result', img)
        
    def inference(self):
        index = int(self.tx_edit_index.text())
        custom_x_test, custom_y_test = np.array([x_test[index]]), np.array([y_test[index]])
        custom_x_test, custom_y_test = [
           torch.from_numpy(custom_x_test.reshape(-1, 1, 28, 28)).float(),
           torch.from_numpy(custom_y_test.astype('long'))
           ]
        custom_dataset = TensorDataset(custom_x_test, custom_y_test)
        custom_loader = DataLoader(dataset=custom_dataset, shuffle=True, batch_size=BATCH_SIZE, **kwargs)
        
        model = torch.load('./F74056328_CNN_model.pth')
        model.eval()

        trainer.test(model, custom_loader)
        
        img = np.reshape(x_test[index, :], (28, 28))
        plt.matshow(img, cmap=plt.get_cmap('gray'))
        plt.title(y_test[index])
        plt.axis('off')
        plt.show()
        global test_validate
        plt.hist(test_validate, bins=np.linspace(0, 10))
        plt.show()

        test_validate = None


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1, self.conv2 = None, None
        self.fc1, self.fc2 = None, None

        self.conv1 = nn.Conv2d(1, 4, 5, stride=1, padding=0, dilation=1, groups=1, bias=False, padding_mode='zeros')
        self.conv2 = nn.Conv2d(4, 8, 5, stride=1, padding=0, dilation=1, groups=1, bias=False, padding_mode='zeros')
        self.fc1 = nn.Linear(800, 96, bias=False)
        self.fc2 = nn.Linear(96, 10, bias=False)

    def forward(self, x):
        out = None

        x = self.conv1(x)
        x = F.max_pool2d(self.conv2(x), (2, 2))
        x = flatten(x)
        x = F.relu(self.fc1(x))
        out = self.fc2(x)

        return out
    
model = CNN()
if OPTIMIZER == 'Adam':
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
elif OPTIMIZER == 'SGD':
    optimizer = torch.optim.SGD(params=model.parameters(),lr=1e-3, momentum=0.9)
else:
    print('Warning: OPTIMIZER is not defined.')

# I took this Trainer from another AI class. Provided by their TAs
# Only modified some parts, so it maybe likely others'
# TA please don't mind
class Trainer:
    def __init__(self, criterion, optimizer, device):
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        
    def train_loop(self, model, train_loader):
        for epoch in range(EPOCHS):
            print("---------------- Epoch {} ----------------".format(epoch))
            self._training_step(model, train_loader, epoch)
    
    def test(self, model, test_loader):
            print("---------------- Testing ----------------")
            self._validate(model, test_loader, 0, state="Testing")
            
    def _training_step(self, model, loader, epoch):
        model.train()
        
        for step, (X, y) in enumerate(loader):
            X, y = X.to(self.device), y.to(self.device)
            
            self.optimizer.zero_grad()
            outs = model(X)
            y = y.long()
            loss = self.criterion(outs, y)

            # special case for section 5.3 in order to record the loss
            if EPOCHS == 1:
                loss_training.append(loss)
            
            if step >= 0 and (step % PRINT_FREQ == 0):
                self._state_logging(outs, y, loss, step, epoch, "Training")
            
            loss.backward()
            self.optimizer.step()
            
    def _validate(self, model, loader, epoch, state="Validate"):
        model.eval()
        outs_list = []
        loss_list = []
        y_list = []
        
        with torch.no_grad():
            for step, (X, y) in enumerate(loader):
                X, y = X.to(self.device), y.to(self.device)
                
                outs = model(X)
                y = y.long()
                global test_validate
                test_validate = y
                loss = self.criterion(outs, y)
                
                y_list.append(y)
                outs_list.append(outs)
                loss_list.append(loss)
            
            y = torch.cat(y_list)
            outs = torch.cat(outs_list)
            loss = torch.mean(torch.stack(loss_list), dim=0)
            self._state_logging(outs, y, loss, step, epoch, state)
                
                
    def _state_logging(self, outs, y, loss, step, epoch, state):
        acc = self._accuracy(outs, y)
        loss_validate.append(loss)
        accuracy_validate.append(acc)
        print("[{:3d}/{}] {} Step {:03d} Loss {:.3f} Acc {:.3f}".format(epoch+1, EPOCHS, state, step, loss, acc))
            
    def _accuracy(self, output, target):
        batch_size = target.size(0)

        pred = output.argmax(1)
        correct = pred.eq(target)
        acc = correct.float().sum(0) / batch_size

        return acc

trainer = Trainer(criterion, optimizer, device)

if __name__ == '__main__':
    main()