################
# Library
################
import numpy as np
import os
import matplotlib.pyplot as plt

import tensorflow as tf

from keras import backend as K
K.set_learning_phase(0) #set learning phase

from keras.datasets import mnist
from keras.models import Model, load_model
from keras.layers import Input, UpSampling2D, AveragePooling2D
from keras.layers import Dense, Dropout, Activation, GlobalAveragePooling2D
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.utils import np_utils
from keras.callbacks import TensorBoard, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator

import sys
sys.path.append('../')

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
        x = Conv2D(96, (3, 3), activation='relu', padding='same', kernel_initializer="glorot_uniform")(input_img)
        x = Conv2D(96, (3, 3), activation='relu', padding='same', kernel_initializer="glorot_uniform")(x)
        x = Conv2D(96, (3, 3), activation='relu', padding='same', kernel_initializer="glorot_uniform")(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Dropout(0.5)(x)
        x = Conv2D(192, (3, 3), activation='relu', padding='same', kernel_initializer="glorot_uniform")(x)
        x = Conv2D(192, (3, 3), activation='relu', padding='same', kernel_initializer="glorot_uniform")(x)
        x = Conv2D(192, (3, 3), activation='relu', padding='same', kernel_initializer="glorot_uniform")(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Dropout(0.5)(x)
        x = Conv2D(192, (3, 3), activation='relu', padding='same', kernel_initializer="glorot_uniform")(x)
        x = Conv2D(192, (1, 1), activation='relu', padding='same', kernel_initializer="glorot_uniform")(x)
        x = Conv2D(10, (1, 1), activation='relu', padding='same', kernel_initializer="glorot_uniform")(x)
        x = GlobalAveragePooling2D()(x)
        # output
        output = Dense(10)(x)
        
        # loss
        def fn(correct, predicted):
            return tf.nn.softmax_cross_entropy_with_logits(labels=correct,
                                                           logits=predicted)
        # classifier
        self.model = Model(input_img, output)
        self.model.compile(optimizer='sgd', loss=fn, metrics=['accuracy'])
        print(self.model.summary())

    def train(self, X, y, X_val, y_val, save_path='models/cifar_classifier'):
        if os.path.exists(save_path):
            self.restore(save_path)
        else:
            # fit generator
            datagen = ImageDataGenerator(width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True)
            datagen.fit(X)
            # fit model
            self.model.fit_generator(datagen.flow(X, y, batch_size=32), \
                                steps_per_epoch=len(X) / 32, epochs=350, \
                                validation_data=(X_val, y_val), \
                                callbacks=[TensorBoard(log_dir='/tmp/classifier')])
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
    cifar = CIFAR10()
    # classifier
    classifier = Classifier(input_shape=cifar.IMG_SHAPE)
    # training
    classifier.train(cifar.X_train, cifar.y_train, cifar.X_test, cifar.y_test)
