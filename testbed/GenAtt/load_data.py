################
# Library
################
from keras.utils import np_utils
from keras.datasets import mnist
from keras.datasets import cifar10
import numpy as np
from skimage.transform import resize






################
# Constant
################













################
# Functions
################
class MNIST:
    def __init__(self):
        # set parameters
        self.IMG_ROW = 28
        self.IMG_COL = 28
        self.IMG_CHA = 1
        self.IMG_CLA = 10
        self.IMG_SHAPE = (self.IMG_ROW, self.IMG_COL, self.IMG_CHA)
        self.IMG_MIN = 0
        self.IMG_MAX = 1
        # loading data
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        # 255 degree to [0,1]
        X_train = X_train.astype('float32') / 255.
        X_test = X_test.astype('float32') / 255.
        # input reshape 
        X_train = X_train.reshape(-1, self.IMG_ROW, self.IMG_COL, self.IMG_CHA)
        X_test = X_test.reshape(-1, self.IMG_ROW, self.IMG_COL, self.IMG_CHA)
        # one hot encoding
        y_train = np_utils.to_categorical(y_train, self.IMG_CLA)
        y_test = np_utils.to_categorical(y_test, self.IMG_CLA)
        # assign to variable
        self.X_train = X_train
        self.y_train = y_train
        self.X_test  = X_test
        self.y_test  = y_test





class MNIST_move:
    def __init__(self, Enlarge=1.5, moving_rate=0.5):
        # set parameters
        self.IMG_ROW = 28
        self.IMG_COL = 28
        self.IMG_CHA = 1
        self.IMG_CLA = 10
        self.IMG_SHAPE = (self.IMG_ROW, self.IMG_COL, self.IMG_CHA)
        self.IMG_MIN = 0
        self.IMG_MAX = 1
        self.Enlarge = Enlarge
        # set background
        self.ImgBoundary = (int(self.IMG_ROW * Enlarge), int(self.IMG_COL * Enlarge), self.IMG_CHA)
        self.Hmax = self.ImgBoundary[0] - self.IMG_ROW
        self.Vmax = self.ImgBoundary[1] - self.IMG_COL
        # loading data
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        # 255 degree to [0,1]
        X_train = X_train.astype('float32') / 255.
        X_test = X_test.astype('float32') / 255.
        # input reshape 
        X_train = X_train.reshape(-1, self.IMG_ROW, self.IMG_COL, self.IMG_CHA)
        X_test = X_test.reshape(-1, self.IMG_ROW, self.IMG_COL, self.IMG_CHA)
        # image moving
        for i in range(X_train.shape[0]):
            if np.random.rand(1) < moving_rate:
                Hbias = np.random.randint(self.Hmax+1)
                Vbias = np.random.randint(self.Vmax+1)
                self.background = np.zeros(self.ImgBoundary, dtype='float32')
                self.background[Hbias:Hbias+self.IMG_ROW, Vbias:Vbias+self.IMG_COL, :] = X_train[i].copy()
                X_train[i] = resize(self.background.copy(), self.IMG_SHAPE)
        for i in range(X_test.shape[0]):
            if np.random.rand(1) < moving_rate:
                Hbias = np.random.randint(self.Hmax+1)
                Vbias = np.random.randint(self.Vmax+1)
                self.background = np.zeros(self.ImgBoundary, dtype='float32')
                self.background[Hbias:Hbias+self.IMG_ROW, Vbias:Vbias+self.IMG_COL, :] = X_test[i].copy()
                X_test[i] = resize(self.background.copy(), self.IMG_SHAPE)
        # one hot encoding
        y_train = np_utils.to_categorical(y_train, self.IMG_CLA)
        y_test = np_utils.to_categorical(y_test, self.IMG_CLA)
        # assign to variable
        self.X_train = X_train
        self.y_train = y_train
        self.X_test  = X_test
        self.y_test  = y_test



        



class CIFAR10:
    def __init__(self):
        # set parameters
        self.IMG_ROW = 32
        self.IMG_COL = 32
        self.IMG_CHA = 3
        self.IMG_CLA = 10
        self.IMG_SHAPE = (self.IMG_ROW, self.IMG_COL, self.IMG_CHA)
        self.IMG_MIN = 0
        self.IMG_MAX = 1
        # loading data
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
        # 255 degree to [0,1]
        X_train = X_train.astype('float32') / 255.
        X_test = X_test.astype('float32') / 255.
        # input reshape 
        X_train = X_train.reshape(-1, self.IMG_ROW, self.IMG_COL, self.IMG_CHA)
        X_test = X_test.reshape(-1, self.IMG_ROW, self.IMG_COL, self.IMG_CHA)
        # one hot encoding
        y_train = np_utils.to_categorical(y_train, self.IMG_CLA)
        y_test = np_utils.to_categorical(y_test, self.IMG_CLA)
        # assign to variable
        self.X_train = X_train
        self.y_train = y_train
        self.X_test  = X_test
        self.y_test  = y_test





