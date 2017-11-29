################
# Library
################
from keras.utils import np_utils
from keras.datasets import mnist
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os
import sys
sys.path.append('/home/guanxiong/Documents/research/Adversarial-Learning/Nov-07/testbed')

from load_data import MNIST
from train_detectors import DetectorI, DetectorII
from Clf.train_classifier import Classifier
import pickle






################
# Constant
################
IMG_ROW = 28
IMG_COL = 28
IMG_CHA = 1
IMG_CLA = 10
IMG_SHAPE = (IMG_ROW, IMG_COL, IMG_CHA)












################
# Functions
################
def gen_figure(img, truth, name):
    fig = plt.figure()
    # original
    plt.subplot(121)
    plt.imshow(truth, cmap='gray', interpolation='none')
    plt.subplot(122)
    plt.imshow(img, cmap='gray', interpolation='none')
    # save fig
    os.makedirs('./success_att/att', exist_ok=True)
    plt.savefig('./success_att/att/'+name)
    plt.close()






################
# main
################
if __name__ == '__main__':
    # load origin examples
    mnist = MNIST()

    # init
    clf = Classifier()
    clf.restore('../Clf/models/classifier')
    dt1 = DetectorI()
    dt1.restore('models/detector1')
    dt2 = DetectorII()
    dt2.restore('models/detector2')

    # get threshold
    test = mnist.X_test
    test_rf1 = dt1.model.predict(test)
    test_rf2 = dt2.model.predict(test)
    rf_error1 = np.sum(np.square(test - test_rf1), axis=(1,2,3))
    rf_error2 = np.sum(np.square(test - test_rf2), axis=(1,2,3))
    thr1 = np.sort(rf_error1, axis=0)[int(rf_error1.shape[0]*0.99)].flatten()[0]
    thr2 = np.sort(rf_error2, axis=0)[int(rf_error2.shape[0]*0.99)].flatten()[0]

    # base line att
    base_att = np.load('../Carlini/results/carlini_l2_0.npy')
    base_truth = np.argmax(mnist.y_train[:128], axis=1)

    # plan classifier
    apred = np.argmax(clf.model.predict(base_att), axis=1)
    rate_plan = len(np.argwhere(apred != base_truth).flatten()) / 128.

    # reformer only
    att_r = dt1.model.predict(base_att)
    rpred = np.argmax(clf.model.predict(att_r), axis=1)
    rate_reformer = len(np.argwhere(rpred != base_truth).flatten()) / 128.

    # detector only
    att_d1 = dt1.model.predict(base_att)
    att_d2 = dt2.model.predict(base_att)
    att_r1 = np.sum(np.square(base_att - att_d1), axis=(1,2,3))
    att_r2 = np.sum(np.square(base_att - att_d2), axis=(1,2,3))
    mask = np.array([(r1 <= thr1) or (r2 <= thr2) for r1, r2 in zip(att_r1, att_r2)])
    rate_detect = 0
    for ap, op, m in zip(apred, base_truth, mask):
        if (m == True) and (ap != op):
            rate_detect += 1
    rate_detect = rate_detect / 128.

    # detector and reformer
    rate_both = 0
    for rp, op, m in zip(rpred, base_truth, mask):
        if (m == True) and (rp != op):
            rate_both += 1
    rate_both = rate_both / 128.

    # visualization
    objects = ('Plan Classifier', 'Reformer Only', 'Detector Only', 'MagNet')
    y_pos = np.arange(len(objects))
    performance = [rate_plan, rate_reformer, rate_detect, rate_both]

    plt.subplot(111)
    plt.bar(y_pos, performance, width=0.35, align='center', alpha=0.5, color='b')
    plt.xticks(y_pos, objects)
    plt.ylabel('Ratio of Successful Attacks')
    plt.title('Attacking Performance on Carlini Examples')
    plt.savefig('att_performance_l')
    plt.close()




    # own att
    own_att = att = np.load('../GenAtt/results/att/att.npy')
    own_truth = np.argmax(clf.model.predict(mnist.X_train[:128]), axis=1)

    # plan classifier
    apred = np.argmax(clf.model.predict(own_att), axis=1)
    rate_plan = len(np.argwhere(apred != own_truth).flatten()) / 128.

    # reformer only
    att_r = dt1.model.predict(own_att)
    rpred = np.argmax(clf.model.predict(att_r), axis=1)
    rate_reformer = len(np.argwhere(rpred != own_truth).flatten()) / 128.

    # detector only
    att_d1 = dt1.model.predict(own_att)
    att_d2 = dt2.model.predict(own_att)
    att_r1 = np.sum(np.square(own_att - att_d1), axis=(1,2,3))
    att_r2 = np.sum(np.square(own_att - att_d2), axis=(1,2,3))
    mask = np.array([(r1 <= thr1) or (r2 <= thr2) for r1, r2 in zip(att_r1, att_r2)])
    rate_detect = 0
    for ap, op, m in zip(apred, own_truth, mask):
        if (m == True) and (ap != op):
            rate_detect += 1
    rate_detect = rate_detect / 128.

    # detector and reformer
    rate_both = 0
    for e, (rp, op, m) in enumerate(zip(rpred, own_truth, mask)):
        if (m == True) and (rp != op):
            rate_both += 1
            gen_figure(own_att[e, :, :, 0].reshape(28,28), mnist.X_train[e, :, :, 0].reshape(28,28), 'Index-'+str(e))
    rate_both = rate_both / 128.
    sys.exit()

    # visualization
    performance = [rate_plan, rate_reformer, rate_detect, rate_both]

    plt.subplot(111)
    plt.bar(y_pos   , performance, width=0.35, align='center', alpha=0.5, color='g', label='Our Attack')
    plt.xticks(y_pos, objects)
    plt.ylabel('Ratio of Successful Attacks')
    plt.title('Attacking Performance on Our Examples')
    plt.savefig('att_performance_r')
    plt.close()





