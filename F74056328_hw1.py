import sys
import cv2
import numpy
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.uic import loadUi

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

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

    def load_image(self):
        img = cv2.imread('./images/dog.bmp')
        cv2.imshow('dog', img)
        print('Height: ' + str(img.shape[0]))
        print('Width: ' + str(img.shape[1]))

    def color_conversion(self):
        img = cv2.imread('./images/color.png')
        img_cc = numpy.zeros_like(img)
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

        cv2.imshow('origin', img)
        cv2.imshow('transform', img_transform)

    def perspective_transformation(self):
        img = cv2.imread('./images/OriginalPerspective.png')
        cv2.imshow('OP', img)

if __name__ == '__main__':
    main()