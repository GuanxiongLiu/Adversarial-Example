################
# Library
################
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import sys
sys.path.append('../testbed')

from load_data import MNIST
from train_models import Autoencoder
from Clf.train_classifier import Classifier






################
# Constant
################












################
# Functions
################
def gen_figure(truth, img, name):
    # original
    plt.subplot(121)
    plt.imshow(truth, cmap='gray', interpolation='none')
    # attack
    plt.subplot(122)
    plt.imshow(img, cmap='gray', interpolation='none')
    # save img
    plt.tight_layout()
    plt.savefig('./results/att/'+name)
    plt.close()








################
# Main
################
if __name__ == '__main__':
    # load results
    status = np.load('./results/status.npy')
    att    = np.load('./results/att.npy')

    # load data
    data = MNIST()

    # load classifier
    classifier = Classifier(data.IMG_SHAPE)
    classifier.restore('../Clf/models/classifier')

    # load autoencoder
    autoencoder = Autoencoder(data.IMG_SHAPE)
    autoencoder.restore('models/autoencoder')

    # get true attack index
    index = np.argwhere(status == True)
    index = index.flatten()

    # container for comparison
    compare = -1 * np.ones((status.shape[0], 2))

    # build folder
    os.makedirs('./results/att', exist_ok=True)

    # get reform error distribution
    ratt = att[index]
    rf_att = autoencoder.model.predict(ratt)
    rf_error = np.sum(np.square(ratt - rf_att), axis=(1,2,3))
    plt.subplot(111)
    plt.hist(rf_error, bins=100, range=(0, 49))
    plt.xlabel('Reconstruction Error')
    plt.ylabel('# of samples')
    plt.title('Histogram of Reconstruction Error in Adversarial Data')
    plt.savefig('./results/att/rf_error')
    plt.close()

    # reform error in test data
    rdata = data.X_test
    rf_data = autoencoder.model.predict(rdata)
    rf_error = np.sum(np.square(rdata - rf_data), axis=(1,2,3))
    plt.subplot(111)
    plt.hist(rf_error, bins=100, range=(0, 49))
    plt.xlabel('Reconstruction Error')
    plt.ylabel('# of samples')
    plt.title('Histogram of Reconstruction Error in Test Data')
    plt.savefig('./results/att/rf_error_test')
    plt.close()

    # form figure
    rf_att = autoencoder.model.predict(att)
    for i in index:
        o = data.X_test[i, :, :, :].reshape(-1, data.IMG_ROW, data.IMG_COL, data.IMG_CHA)
        a = att[i, :, :, :].reshape(-1, data.IMG_ROW, data.IMG_COL, data.IMG_CHA)
        compare[i, :] = [np.argmax(classifier.model.predict(o)), np.argmax(classifier.model.predict(a))]
        gen_figure(o.reshape(28, 28), a.reshape(28, 28), \
                    ro.reshape(28, 28), ra.reshape(28, 28), 'Index-'+str(i))


    # save result
    np.save('./results/att/att.npy', att)
    np.save('./results/att/status.npy', status)
    np.save('./results/att/pred_compare.npy', compare)
