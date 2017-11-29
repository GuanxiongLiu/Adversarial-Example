################
# Library
################
import numpy as np
from scipy.stats import entropy
from numpy.linalg import norm
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
from keras import optimizers

import sys
sys.path.append('/home/guanxiong/Documents/research/Adversarial-Learning/Nov-07/testbed')

from GenAtt.load_data import CIFAR10






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
        x = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
        x = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
        x = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
        x = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
        # output
        output = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
        # classifier
        self.model = Model(input_img, output)
        self.model.compile(optimizer='adam', loss='mean_absolute_error', metrics=['mean_absolute_error'])

    def train(self, X, y, X_val, y_val, save_path='models/cifar_detector1'):
        if os.path.exists(save_path):
            self.restore(save_path)
        else:
            # add noise
            X = X + np.random.normal(size=X.shape, scale=0.025)
            X_val = X_val + np.random.normal(size=X_val.shape, scale=0.025)
            # fit model
            self.model.fit(X, y, epochs = 400, batch_size = 256,\
                           shuffle = True, validation_data = (X_val, y_val),\
                           callbacks = [TensorBoard(log_dir = '/tmp/detector1')])
        # save model
        self.model.save(save_path)

    def restore(self, path):
        self.model.load_weights(path)


class DetectorII:
    def __init__(self, tempreture):
        self.tempreture = tempreture

    def softmax(self, x):
        x = x / self.tempreture
        return np.exp(x-np.mean(x)) / np.sum(np.exp(x-np.mean(x)), axis=0)

    def JSD(self, P, Q):
        P = self.softmax(P)
        Q = self.softmax(Q)
        _P = P / norm(P, ord=1)
        _Q = Q / norm(Q, ord=1)
        _M = 0.5 * (_P + _Q)
        return 0.5 * (entropy(_P, _M) + entropy(_Q, _M))






################
# Main
################
if __name__ == '__main__':
    # init
    cifar = CIFAR10()
    detector1 = DetectorI(input_shape=cifar.IMG_SHAPE)
    #detector2 = DetectorII(input_shape=cifar.IMG_SHAPE)
    # classifier
    detector1.train(cifar.X_train, cifar.X_train, cifar.X_test, cifar.X_test, save_path='models/cifar_detector1')
    # autoencoder
    #detector2.train(mnist_move.X_train, mnist_move.X_train, mnist_move.X_test, mnist_move.X_test, save_path='models/detector2_move')
