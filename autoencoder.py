#########################
# Library
#########################
import numpy as np
import os
import matplotlib.pyplot as plt

from keras import backend as K
from keras.datasets import mnist
from keras.models import Model, load_model
from keras.layers import Input, UpSampling2D
from keras.layers import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.utils import np_utils
from keras.callbacks import TensorBoard




#########################
# Pre-defined Functions
#########################
IMG_ROW = 28
IMG_COL = 28
IMG_CHA = 1
IMG_CLA = 10
IMG_SHAPE = (IMG_ROW, IMG_COL, IMG_CHA)



#########################
# Main Function
#########################
if __name__ == '__main__':
    print('========= Loading Dataset =========')
    (X_train, _), (X_test, _) = mnist.load_data()
   
    # 255 degree to [0,1] 
    X_train = X_train.astype('float32') / 255.
    X_test = X_test.astype('float32') / 255.
   
    # input reshape 
    X_train = X_train.reshape(-1, IMG_ROW, IMG_COL, IMG_CHA)
    X_test = X_test.reshape(-1, IMG_ROW, IMG_COL, IMG_CHA)
    


    if os.path.isfile('./autoencoder.h5'):
        print('========= Loading Model =========')
        autoencoder = load_model('./autoencoder.h5')
    else:



        print('========= Building Model =========')
        # input
        input_img = Input(shape=(IMG_ROW, IMG_COL, IMG_CHA))
    
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
        autoencoder = Model(input_img, decoded)
        autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    
    
    
        print('========= Training Model =========')
        autoencoder.fit(X_train, X_train,\
                        epochs = 50,\
                        batch_size = 128,\
                        shuffle = True,\
                        validation_data = (X_test, X_test),\
                        callbacks = [TensorBoard(log_dir = '/tmp/autoencoder')])



        print('========= Saving Model =========')
        autoencoder.save('./autoencoder.h5')



    print('========= Visualization =========')
    np.random.shuffle(X_test)
    decoded_imgs = autoencoder.predict(X_test)

    n = 10
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(X_test[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    
        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()









