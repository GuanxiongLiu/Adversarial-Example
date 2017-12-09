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

    # load classifier and detectors
    classifier = Classifier(input_shape=data.IMG_SHAPE)
    classifier.restore('../Clf/models/cifar_classifier')
    dt1 = DetectorI(input_shape=data.IMG_SHAPE)
    dt1.restore('models/cifar_detector1')
    dt2 = DetectorII(tempreture=10.)
    dt3 = DetectorII(tempreture=40.)

    # prediction on original data
    #opred = np.argmax(classifier.model.predict(test[:128]), axis=1)

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

    # load attack and groundtruth
    att = np.load(sys.argv[1])
    att_name = sys.argv[2]
    groundtruth = data.y_test[:128]

    # detection and prediction
    apred = np.argmax(classifier.model.predict(att), axis=1)
    rpred = np.argmax(classifier.model.predict(dt1.model.predict(att)), axis=1)

    # evaluation
    rate_clf = 0
    rate_ref = 0
    rate_det = 0
    rate_magnet = 0
    rf_a = dt1.model.predict(att)
    a_logit = classifier.model.predict(att)
    rf_a_logit = classifier.model.predict(rf_a)
    for a, rf, logit1, logit2, gt, ap, rp in zip(att, rf_a, a_logit, rf_a_logit, groundtruth, apred, rpred):
        # calculate detector errors
        dt1_e = np.sum(np.square(a - rf), axis=(0,1,2))
        dt2_e = dt2.JSD(logit1, logit2)
        dt3_e = dt3.JSD(logit1, logit2)
        # plain classifier
        if ap != np.argmax(gt):
            rate_clf += 1
        # reformer only
        if rp != np.argmax(gt):
            rate_ref += 1
        # detector only
        if (dt1_e <= rf_thr) and (dt2_e <= jsd2_thr) and (dt3_e <= jsd3_thr) and (ap != np.argmax(gt)):
            rate_det += 1
        # magnet
        if (dt1_e <= rf_thr) and (dt2_e <= jsd2_thr) and (dt3_e <= jsd3_thr) and (rp != np.argmax(gt)):
            rate_magnet += 1

    # print evaluation results
    total_n = att.shape[0]
    print("The accuracy of plain classifier is %f" % ((total_n - rate_clf)/total_n))
    print("The accuracy of reformer only classifier is %f" % ((total_n - rate_ref)/total_n))
    print("The accuracy of detector only classifier is %f" % ((total_n - rate_det)/total_n))
    print("The accuracy of MagNet classifier is %f" % ((total_n - rate_magnet)/total_n))

    # generate bar figure
    objects = ('Plain Classifier', 'Reformer Only', 'Detector Only', 'MagNet')
    y_pos = np.arange(len(objects))
    performance = [((total_n - rate_clf)/total_n), 
                    ((total_n - rate_ref)/total_n), 
                    ((total_n - rate_det)/total_n), 
                    ((total_n - rate_magnet)/total_n)]
    plt.subplot(111)
    plt.bar(y_pos, performance, width=0.35, align='center', alpha=0.5, color='g')
    plt.ylim(ymax=1)
    plt.xticks(y_pos, objects)
    plt.ylabel('Prediction Accuracy')
    plt.title('Attacking Performance on ' + att_name)
    plt.savefig('img/' + sys.argv[3])
    plt.close()
