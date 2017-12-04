################
# Library
################
import numpy as np
import os
import sys
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
sys.path.append('../testbed')

from GenAtt.load_data import MNIST, MNIST_move, CIFAR10






################
# Constant
################












################
# Functions
################
class Autoencoder:
    def __init__(self, input_shape, session=None):
        # input
        input_img = Input(shape=input_shape)
    
        # encoder
        x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        encoded = MaxPooling2D((2, 2), padding='same')(x)
        
        # decoder
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(16, (3, 3), activation='relu')(x)
        x = UpSampling2D((2, 2))(x)
        decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    
        # autoencoder
        self.model = Model(input_img, decoded)
        self.model.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['mean_squared_error'])

    def train(self, X, y, X_val, y_val, save_path='models/mnist_autoencoder'):
        if os.path.exists(save_path):
            self.restore(save_path)
        else:
            self.model.fit(X, y, epochs = 50, batch_size = 128, shuffle = True,\
                            validation_data = (X_val, y_val), callbacks = [TensorBoard(log_dir = '/tmp/autoencoder')])
        # save model
        self.model.save(save_path)

    def restore(self, path):
        self.model.load_weights(path)






def visualize(data, autoencoder, noise_factor=5):
    index = int(np.random.rand(1) * data.X_test.shape[0])
    img = data.X_test[index].reshape(-1,data.IMG_ROW, data.IMG_COL, data.IMG_CHA)
    dim = str(2) + str(noise_factor+1)
    if img.shape[3] == 1:
        for i in range(1,2+noise_factor):
            if i == 1:
                plt.subplot(2, noise_factor+1, i)
                plt.imshow(img.reshape(data.IMG_ROW, data.IMG_COL), cmap='gray', interpolation='none')
                plt.subplot(2, noise_factor+1, i+noise_factor+1)
                plt.imshow(autoencoder.model.predict(img).reshape(data.IMG_ROW, data.IMG_COL), cmap='gray', interpolation='none')
            else:
                noise = img + np.random.normal(size=img.shape, scale=0.1*(i-1))
                plt.subplot(2, noise_factor+1, i)
                plt.imshow(noise.reshape(data.IMG_ROW, data.IMG_COL), cmap='gray', interpolation='none')
                plt.subplot(2, noise_factor+1, i+noise_factor+1)
                plt.imshow(autoencoder.model.predict(noise).reshape(data.IMG_ROW, data.IMG_COL), cmap='gray', interpolation='none')
    else:
        for i in range(1,2+noise_factor):
            if i == 1:
                plt.subplot(2, noise_factor+1, i)
                plt.imshow(img.reshape(data.IMG_ROW, data.IMG_COL, data.IMG_CHA), interpolation='none')
                plt.subplot(2, noise_factor+1, i+noise_factor+1)
                plt.imshow(autoencoder.model.predict(img).reshape(data.IMG_ROW, data.IMG_COL, data.IMG_CHA), interpolation='none')
            else:
                noise = img + np.random.normal(size=img.shape, scale=0.1*(i-1))
                plt.subplot(2, noise_factor+1, i)
                plt.imshow(noise.reshape(data.IMG_ROW, data.IMG_COL, data.IMG_CHA), interpolation='none')
                plt.subplot(2, noise_factor+1, i+noise_factor+1)
                plt.imshow(autoencoder.model.predict(noise).reshape(data.IMG_ROW, data.IMG_COL, data.IMG_CHA), interpolation='none')
    plt.tight_layout()
    plt.savefig('test_visualization')
    plt.close()




def reform_test_mnist(data, autoencoder):
    index = int(np.random.rand(1) * data.X_test.shape[0])
    img = data.X_test[index].reshape(-1,data.IMG_ROW, data.IMG_COL, data.IMG_CHA)
    # horizontal test
    for i in range(1,6):
        if i == 1:
            plt.subplot(2, 5, i)
            plt.imshow(img.reshape(data.IMG_ROW, data.IMG_COL), cmap='gray', interpolation='none')
            plt.subplot(2, 5, i+5)
            plt.imshow(autoencoder.model.predict(img).reshape(data.IMG_ROW, data.IMG_COL), cmap='gray', interpolation='none')
        else:
            t_img = img.copy()
            t_img[0, int((i-2)*(data.IMG_ROW/4.)), :, :] = 0.99
            plt.subplot(2, 5, i)
            plt.imshow(t_img.reshape(data.IMG_ROW, data.IMG_COL), cmap='gray', interpolation='none')
            plt.subplot(2, 5, i+5)
            plt.imshow(autoencoder.model.predict(t_img).reshape(data.IMG_ROW, data.IMG_COL), cmap='gray', interpolation='none')
    plt.tight_layout()
    plt.savefig('reform_test_horizontal')
    plt.close()
    # vertical test
    for i in range(1,6):
        if i == 1:
            plt.subplot(2, 5, i)
            plt.imshow(img.reshape(data.IMG_ROW, data.IMG_COL), cmap='gray', interpolation='none')
            plt.subplot(2, 5, i+5)
            plt.imshow(autoencoder.model.predict(img).reshape(data.IMG_ROW, data.IMG_COL), cmap='gray', interpolation='none')
        else:
            t_img = img.copy()
            t_img[0, :, int((i-2)*(data.IMG_COL/4.)), :] = 0.99
            plt.subplot(2, 5, i)
            plt.imshow(t_img.reshape(data.IMG_ROW, data.IMG_COL), cmap='gray', interpolation='none')
            plt.subplot(2, 5, i+5)
            plt.imshow(autoencoder.model.predict(t_img).reshape(data.IMG_ROW, data.IMG_COL), cmap='gray', interpolation='none')
    plt.tight_layout()
    plt.savefig('reform_test_vertical')
    plt.close()





################
# Main
################
if __name__ == '__main__':
    # init
    mnist = MNIST()
    mnist_move = MNIST_move()
    #autoencoder = Autoencoder(input_shape=mnist.IMG_SHAPE)
    autoencoder = Autoencoder(input_shape=mnist_move.IMG_SHAPE)
    # adding noise to train
    #train_noise = np.random.normal(size=mnist.X_train.shape, scale=0.5)
    #train_in = train_noise + mnist.X_train
    # adding noise to test
    #test_noise = np.random.normal(size=mnist.X_test.shape, scale=0.5)
    #test_in = test_noise + mnist.X_test
    # autoencoder
    #autoencoder.train(mnist.X_train, mnist.X_train, mnist.X_test, mnist.X_test)
    autoencoder.train(mnist_move.X_train, mnist_move.X_train, mnist_move.X_test, mnist_move.X_test, save_path='models/mnist_autoencoder_move')

    # test visualization
    visualize(mnist, autoencoder)
    reform_test_mnist(mnist, autoencoder)
