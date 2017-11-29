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
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
   
    # 255 degree to [0,1] 
    X_train = X_train.astype('float32') / 255.
    X_test = X_test.astype('float32') / 255.
   
    # input reshape 
    X_train = X_train.reshape(-1, IMG_ROW, IMG_COL, IMG_CHA)
    X_test = X_test.reshape(-1, IMG_ROW, IMG_COL, IMG_CHA)

    # one hot encoding
    y_train = np_utils.to_categorical(y_train, IMG_CLA)
    y_test = np_utils.to_categorical(y_test, IMG_CLA)
    


    if os.path.isfile('./classifier.h5'):
        print('========= Loading Model =========')
        autoencoder = load_model('./classifier.h5')
    else:



        print('========= Building Model =========')
        # input
        input_img = Input(shape=(IMG_ROW, IMG_COL, IMG_CHA))
    
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
        classifier = Model(input_img, output)
        classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    
        print('========= Training Model =========')
        classifier.fit(X_train, y_train,\
                        epochs = 50,\
                        batch_size = 128,\
                        shuffle = True,\
                        validation_data = (X_test, y_test),\
                        callbacks = [TensorBoard(log_dir = '/tmp/classifier')])



        print('========= Saving Model =========')
        classifier.save('./classifier.h5')



