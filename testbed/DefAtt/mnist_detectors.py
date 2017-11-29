################
# Library
################
import numpy as np
import os
import matplotlib.pyplot as plt

from keras import backend as K
from keras.datasets import mnist
from keras.models import Model, load_model
from keras.layers import Input, UpSampling2D, AveragePooling2D
from keras.layers import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.utils import np_utils
from keras.callbacks import TensorBoard

import sys
sys.path.append('/home/guanxiong/Documents/research/Adversarial-Learning/Nov-07/testbed')

from GenAtt.load_data import MNIST, MNIST_move






################
# Constant
################












################
# Functions
################
class DetectorI:
    def __init__(self, input_shape, session=None):
        # input
        input_img = Input(shape=input_shape)
        # hidden layers
        x = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(input_img)
        x = AveragePooling2D((2, 2), padding="same")(x)
        x = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
        x = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
        # output
        output = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
        # classifier
        self.model = Model(input_img, output)
        self.model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])

    def train(self, X, y, X_val, y_val, save_path='models/mnist_detector1'):
        if os.path.exists(save_path):
            self.restore(save_path)
        else:
            # fit model
            self.model.fit(X, y, epochs = 100, batch_size = 256,\
                           shuffle = True, validation_data = (X_val, y_val),\
                           callbacks = [TensorBoard(log_dir = '/tmp/detector1')])
        # save model
        self.model.save(save_path)

    def restore(self, path):
        self.model.load_weights(path)


class DetectorII:
    def __init__(self, input_shape, session=None):
        # input
        input_img = Input(shape=input_shape)
        # hidden layers
        x = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(input_img)
        x = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
        # output
        output = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
        # classifier
        self.model = Model(input_img, output)
        self.model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])

    def train(self, X, y, X_val, y_val, save_path='models/mnist_detector2'):
        if os.path.exists(save_path):
            self.restore(save_path)
        else:
            # fit model
            self.model.fit(X, y, epochs = 100, batch_size = 256,\
                           shuffle = True, validation_data = (X_val, y_val),\
                           callbacks = [TensorBoard(log_dir = '/tmp/detector2')])
        # save model
        self.model.save(save_path)

    def restore(self, path):
        self.model.load_weights(path)






################
# Main
################
if __name__ == '__main__':
    # init
    mnist = MNIST()
    mnist_move = MNIST_move()
    detector1 = DetectorI(input_shape=mnist_move.IMG_SHAPE)
    detector2 = DetectorII(input_shape=mnist_move.IMG_SHAPE)
    # classifier
    detector1.train(mnist_move.X_train, mnist_move.X_train, mnist_move.X_test, mnist_move.X_test, save_path='models/detector1_move')
    # autoencoder
    detector2.train(mnist_move.X_train, mnist_move.X_train, mnist_move.X_test, mnist_move.X_test, save_path='models/detector2_move')
