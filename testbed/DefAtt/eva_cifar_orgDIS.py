################
# Library
################
import tensorflow as tf
from keras import backend as K
from keras.utils import np_utils
from keras.datasets import mnist
K.set_learning_phase(0) #set learning phase
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('../')

from GenAtt.load_data import CIFAR10
from Carlini.setup_cifar import CIFARModel






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
    with tf.Session() as sess:
        # load original data
        data = CIFAR10()
        test = data.X_test

        # load classifier and detectors
        classifier = CIFARModel('../Carlini/models/cifar', sess)
        classifier_dis = CIFARModel('../Carlini/models/cifar-distilled-100', sess)

        # load attack
        att_org = np.load(sys.argv[1])
        att_dis = np.load(sys.argv[2])
        att_name = sys.argv[3]
        groundtruth = data.y_test[:128]

        # detection and prediction
        apred = np.argmax(classifier.model.predict(att_org), axis=1)
        dpred = np.argmax(classifier_dis.model.predict(att_dis), axis=1)

        # evaluation
        rate_clf = 0
        rate_dis = 0
        for ap,dp,gt in zip(apred, dpred, groundtruth):
            # plain classifier
            if ap != np.argmax(gt):
                rate_clf += 1
            # distilled classifier
            if dp != np.argmax(gt):
                rate_dis += 1

        # print evaluation results
        total_n = att_org.shape[0]
        print("The accuracy of plain classifier is %f" % ((total_n - rate_clf)/total_n))
        print("The accuracy of distilled classifier is %f" % ((total_n - rate_dis)/total_n))

        # generate bar figure
        objects = ('Plain Classifier', 'Distilled Classifier')
        y_pos = np.arange(len(objects))
        performance = [((total_n - rate_clf)/total_n), 
                        ((total_n - rate_dis)/total_n)]
        plt.subplot(111)
        plt.bar(y_pos, performance, width=0.35, align='center', alpha=0.5, color='g')
        plt.ylim(ymax=1)
        plt.xticks(y_pos, objects)
        plt.ylabel('Prediction Accuracy')
        plt.title('Attacking Performance on ' + att_name)
        plt.savefig('img/' + sys.argv[4])
        plt.close()
