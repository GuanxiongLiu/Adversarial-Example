################
# Library
################
from keras.utils import np_utils
from keras.datasets import mnist
from keras import backend as K
K.set_learning_phase(0) #set learning phase
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
sys.path.append('../')

from GenAtt.load_data import CIFAR10
from cifar_detector import DetectorI, DetectorII
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

    # load classifier and detectors
    classifier = Classifier(input_shape=data.IMG_SHAPE)
    classifier.restore('../Clf/models/cifar_classifier')
    dt1 = DetectorI(input_shape=data.IMG_SHAPE)
    dt1.restore('models/cifar_detector1')
    dt2 = DetectorII(tempreture=10.)
    dt3 = DetectorII(tempreture=40.)

    # prediction on original data
    opred = classifier.model.predict(test[:128])

    # calculate threshold from test
    rf_test = dt1.model.predict(test)
    rf_error = np.sum(np.square(test - rf_test), axis=(1,2,3))
    rf_thr = np.sort(rf_error, axis=0)[int(rf_error.shape[0]*0.9)].flatten()[0]
    jsd2, jsd3 = [], []
    d_logit = classifier.model.predict(test)
    rf_logit = classifier.model.predict(rf_test)
    for d, rf in zip(d_logit, rf_logit):
        jsd2.append(dt2.JSD(d, rf))
        jsd3.append(dt3.JSD(d, rf))
    jsd2 = np.array(jsd2)
    jsd2_thr = np.sort(jsd2, axis=0)[int(jsd2.shape[0]*0.9)].flatten()[0]
    jsd3 = np.array(jsd3)
    jsd3_thr = np.sort(jsd3, axis=0)[int(jsd3.shape[0]*0.9)].flatten()[0]

    # load attack
    att = np.load(sys.argv[1])
    att_name = sys.argv[2]

    # detection and prediction
    apred = np.argmax(classifier.model.predict(att), axis=1)
    rpred = np.argmax(classifier.model.predict(dt1.model.predict(att)), axis=1)

    # evaluation
    index = 0
    directory = ('img/%s/' % str(att_name))
    if not os.path.exists(directory):
        os.makedirs(directory)
    rf_a = dt1.model.predict(att)
    a_logit = classifier.model.predict(att)
    rf_a_logit = classifier.model.predict(rf_a)
    for a, rf, logit1, logit2, op, ap, rp in zip(att, rf_a, a_logit, rf_a_logit, opred, apred, rpred):
        # calculate detector errors
        dt1_e = np.sum(np.square(a - rf), axis=(0,1,2))
        dt2_e = dt2.JSD(logit1, logit2)
        dt3_e = dt3.JSD(logit1, logit2)
        # magnet
        if (dt1_e <= rf_thr) and (dt2_e <= jsd2_thr) and (dt3_e <= jsd3_thr) and (rp != np.argmax(op)):
            plt.figure(figsize = (3,6))
            plt.subplot(121)
            plt.imshow(test[index].reshape(32,32,3), interpolation='nearest')
            plt.subplot(122)
            plt.imshow(a.reshape(32,32,3), interpolation='nearest')
            plt.savefig(directory + str(index))
            plt.close()
        # update index
        index += 1
