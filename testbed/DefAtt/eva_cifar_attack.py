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
sys.path.append('/home/guanxiong/Documents/research/Adversarial-Learning/Nov-07/testbed')

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

    # calculate threshold from test
    rf_test = dt1.model.predict(test)
    rf_error = np.sum(np.square(test - rf_test), axis=(1,2,3))
    rf_thr = np.sort(rf_error, axis=0)[int(rf_error.shape[0]*0.9)].flatten()[0]
    jsd2, jsd3 = [], []
    for d, rf in zip(test, rf_test):
        d_logit = classifier.model.predict(d.reshape(-1,32,32,3)).flatten()
        rf_logit = classifier.model.predict(rf.reshape(-1,32,32,3)).flatten()
        jsd2.append(dt2.JSD(d_logit, rf_logit))
        jsd3.append(dt3.JSD(d_logit, rf_logit))
    jsd2 = np.array(jsd2)
    jsd2_thr = np.sort(jsd2, axis=0)[int(jsd2.shape[0]*0.9)].flatten()[0]
    jsd3 = np.array(jsd3)
    jsd3_thr = np.sort(jsd3, axis=0)[int(jsd3.shape[0]*0.9)].flatten()[0]

    # load attack
    att = np.load('../Carlini/results/cifar_carlini_l2_10.npy')
    att_name = 'Carlini Attack'

    # detection and prediction
    apred = np.argmax(classifier.model.predict(att), axis=1)
    rpred = np.argmax(classifier.model.predict(dt1.model.predict(att)), axis=1)

    # evaluation
    rate_clf = 0
    rate_ref = 0
    rate_det = 0
    rate_magnet = 0
    for a, gt, ap, rp in zip(att, groundtruth, apred, rpred):
        # calculate detector errors
        rf_a = dt1.model.predict(a.reshape(-1,32,32,3))
        a_logit = classifier.model.predict(a.reshape(-1,32,32,3)).flatten()
        rf_a_logit = classifier.model.predict(rf_a.reshape(-1,32,32,3)).flatten()
        dt1_e = np.sum(np.square(a - rf_a), axis=(1,2,3))
        dt2_e = dt2.JSD(a_logit, rf_a_logit)
        dt3_e = dt3.JSD(a_logit, rf_a_logit)
        # plain classifier
        if np.argmax(ap) != np.argmax(gt):
            rate_clf += 1
        # reformer only
        if np.argmax(rp) != np.argmax(gt):
            rate_ref += 1
        # detector only
        if (dt1_e <= rf_thr) and (dt2_e <= jsd2_thr) and (dt3_e <= jsd3_thr) and (np.argmax(ap) != np.argmax(gt)):
            rate_det += 1
        # magnet
        if (dt1_e <= rf_thr) and (dt2_e <= jsd2_thr) and (dt3_e <= jsd3_thr) and (np.argmax(rp) != np.argmax(gt)):
            rate_magnet += 1

    # print evaluation results
    print("The attacking rate of plain classifier is %f" % (rate_clf/att.shape[0]))
    print("The attacking rate of reformer only classifier is %f" % (rate_ref/att.shape[0]))
    print("The attacking rate of detector only classifier is %f" % (rate_det/att.shape[0]))
    print("The attacking rate of MagNet classifier is %f" % (rate_magnet/att.shape[0]))

    # generate bar figure
    objects = ('Plain Classifier', 'Reformer Only', 'Detector Only', 'MagNet')
    y_pos = np.arange(len(objects))
    performance = [rate_clf, rate_ref, rate_det, rate_magnet]
    plt.subplot(111)
    plt.bar(y_pos, performance, width=0.35, align='center', alpha=0.5, color='g')
    plt.xticks(y_pos, objects)
    plt.ylabel('Ratio of Successful Attacks')
    plt.title('Attacking Performance on ' + att_name)
    plt.savefig('img/att_performance_' + att_name)
    plt.close()
