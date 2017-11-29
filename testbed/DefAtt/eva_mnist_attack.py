################
# Library
################
from keras.utils import np_utils
from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('/home/guanxiong/Documents/research/Adversarial-Learning/Nov-07/testbed')

from GenAtt.load_data import MNIST, MNIST_move
from train_detectors import DetectorI, DetectorII
from Clf.train_classifier import Classifier






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
    # load adversarial examples
    att = np.load('../GenAtt/results/mnist_move_img/att.npy')
    status = np.load('../GenAtt/results/mnist_move_img/status.npy')
    compare = np.load('../GenAtt/results/mnist_move_img/pred_compare.npy')

    # init
    mnist = MNIST()
    mnist_move = MNIST_move()
    clf = Classifier(input_shape=mnist_move.IMG_SHAPE)
    clf.restore('../Clf/models/classifier_move')
    dt1 = DetectorI(input_shape=mnist_move.IMG_SHAPE)
    dt1.restore('models/detector1_move')
    dt2 = DetectorII(input_shape=mnist_move.IMG_SHAPE)
    dt2.restore('models/detector2_move')

    # get threshold
    test = mnist.X_test
    test_rf1 = dt1.model.predict(test)
    test_rf2 = dt2.model.predict(test)
    rf_error1 = np.sum(np.square(test - test_rf1), axis=(1,2,3))
    rf_error2 = np.sum(np.square(test - test_rf2), axis=(1,2,3))
    thr1 = np.sort(rf_error1, axis=0)[int(rf_error1.shape[0]*0.9)].flatten()[0]
    thr2 = np.sort(rf_error2, axis=0)[int(rf_error2.shape[0]*0.9)].flatten()[0]

    # carlini att
    base_att = np.load('../Carlini/results/carlini_l2_0.npy')
    #att = base_att

    # plan classifier
    opred = np.argmax(clf.model.predict(mnist.X_test[:128]), axis=1)
    apred = np.argmax(clf.model.predict(att), axis=1)
    rate_plan = len(np.argwhere(apred != opred).flatten()) / 128.

    # reformer only
    att_r = dt1.model.predict(att)
    rpred = np.argmax(clf.model.predict(att_r), axis=1)
    rate_reformer = len(np.argwhere(rpred != opred).flatten()) / 128.

    # detector only
    att_d1 = dt1.model.predict(att)
    att_d2 = dt2.model.predict(att)
    att_r1 = np.sum(np.square(att - att_d1), axis=(1,2,3))
    att_r2 = np.sum(np.square(att - att_d2), axis=(1,2,3))
    mask = np.array([(r1 <= thr1) or (r2 <= thr2) for r1, r2 in zip(att_r1, att_r2)])
    rate_detect = 0
    for ap, op, m in zip(apred, opred, mask):
        if (m == True) and (ap != op):
            rate_detect += 1
    rate_detect = rate_detect / 128.

    # detector and reformer
    rate_both = 0
    for rp, op, m in zip(apred, opred, mask):
        if (m == True) and (rp != op):
            rate_both += 1
    rate_both = rate_both / 128.

    # print result
    print(len(np.argwhere(mask == True).flatten()))
    print("The attacking rate of plan classifier is %f" % rate_plan)
    print("The attacking rate of reformer only classifier is %f" % rate_reformer)
    print("The attacking rate of detector only classifier is %f" % rate_detect)
    print("The attacking rate of MagNet classifier is %f" % rate_both)

    # visualization
    objects = ('Plan Classifier', 'Reformer Only', 'Detector Only', 'MagNet')
    y_pos = np.arange(len(objects))
    performance = [rate_plan, rate_reformer, rate_detect, rate_both]

    plt.subplot(111)
    plt.bar(y_pos, performance, width=0.35, align='center', alpha=0.5, color='g')
    plt.xticks(y_pos, objects)
    plt.ylabel('Ratio of Successful Attacks')
    plt.title('Attacking Performance on Our Examples')
    plt.savefig('att_performance_own')
    plt.close()





