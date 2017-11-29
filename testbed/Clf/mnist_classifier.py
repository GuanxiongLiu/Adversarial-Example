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
from keras.layers import Dense, Dropout, Activation, GlobalAveragePooling2D
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.utils import np_utils
from keras.callbacks import TensorBoard

import sys
sys.path.append('/home/guanxiong/Documents/research/Adversarial-Learning/Nov-07/testbed')

from GenAtt.load_data import MNIST, MNIST_move, CIFAR10






################
# Constant
################












################
# Functions
################
class Classifier:
    def __init__(self, input_shape, session=None):
        # input
        input_img = Input(shape=input_shape)
        # hidden layers
        x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        # output
        output = Dense(10, activation='softmax')(x)
        # classifier
        self.model = Model(input_img, output)
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def train(self, X, y, X_val, y_val, save_path='models/mnist_classifier'):
        if os.path.exists(save_path):
            self.restore(save_path)
        else:
            # fit model
            self.model.fit(X, y, epochs = 100, batch_size = 128,\
                            shuffle = True, validation_data = (X_val, y_val),\
                            callbacks = [TensorBoard(log_dir = '/tmp/classifier')])
        # save model
        self.model.save(save_path)

    def restore(self, path):
        self.model.load_weights(path)

    def predict(self, data):
        return self.model(data)









################
# Main
################
if __name__ == '__main__':
    # init
    mnist = MNIST()
    mnist_move = MNIST_move()
    # classifier
    classifier = Classifier(input_shape=mnist_move.IMG_SHAPE)
    # training
    classifier.train(mnist_move.X_train, mnist_move.y_train, mnist_move.X_test, mnist_move.y_test, save_path='models/classifier_move')
