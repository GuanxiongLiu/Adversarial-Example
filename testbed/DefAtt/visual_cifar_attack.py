################
# Library
################
from keras.utils import np_utils
from keras.datasets import mnist
from keras import backend as K
K.set_learning_phase(0) #set learning phase
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('../')

from GenAtt.load_data import CIFAR10
from Clf.cifar_classifier import Classifier






################
# Constant
################












################
# Functions
################






################
# main
################
if __name__ == '__main__':
    # load original data
    data = CIFAR10()
    test = data.X_test
    groundtruth = data.y_test[:128]

    # load attack
    att_base = np.load(sys.argv[1])
    att_self = np.load(sys.argv[2])

    # random index
    index = np.random.randint(low=0, high=128, size=4)

    # generate figure
    plt.figure()
    for i in range(3):
        for j in range(4):
            plt.subplot(3, 4, i*4+j+1)
            if i == 0:
                plt.title('Image #' + str(index[j]))
                img = test[index[j]].reshape(32,32,3)
            if i == 1:
                img = att_base[index[j]].reshape(32,32,3)
            if i == 2:
                img = att_self[index[j]].reshape(32,32,3)
            plt.imshow(img)
    plt.savefig('img/'+sys.argv[3])
    plt.close()
