################
# Library
################
import numpy as np
from skimage.transform import resize
import sys
sys.path.append('/home/guanxiong/Documents/research/Adversarial-Learning/Nov-07/testbed')

from load_data import MNIST
from train_models import Autoencoder
from Clf.train_classifier import Classifier
import tensorflow as tf






################
# Constant
################
Enlarge = 1.5
B_SIZE  = 128
GEN_NUM = 1












################
# Functions
################
class MoveImg:
    def __init__(self, data):
        # load data
        self.data = data
        self.classifier = Classifier(input_shape=data.IMG_SHAPE)
        self.classifier.restore('../Clf/models/classifier')
        self.autoencoder = Autoencoder(input_shape=data.IMG_SHAPE)
        self.autoencoder.restore('models/autoencoder')

        # set parameter
        self.ImgBoundary = (int(self.data.IMG_ROW * Enlarge), int(self.data.IMG_COL * Enlarge), self.data.IMG_CHA)
        self.Hmax = self.ImgBoundary[0] - self.data.IMG_ROW
        self.Vmax = self.ImgBoundary[1] - self.data.IMG_COL

    def launch(self):
        X = self.data.X_test
        Y = self.data.y_test
        final_att = []
        final_status = []
        for i in range(0, min(X.shape[0], B_SIZE*GEN_NUM), B_SIZE):
            # get current batch data
            batch_X = X[i:i+B_SIZE]
            batch_Y = Y[i:i+B_SIZE]

            # setup att holder
            current_att = np.zeros(batch_X.shape)
            current_status = []

            # loop through to move image
            for j in range(B_SIZE):
                img = batch_X[j]
                # random horizontal and vertical bias
                #Hbias = np.random.randint(self.Hmax+1)
                Hbias = 0
                #Vbias = np.random.randint(self.Vmax+1)
                Vbias = 0
                # paste image to background, resize and copy over
                self.background = np.zeros(self.ImgBoundary, dtype='float32')
                self.background[Hbias:Hbias+self.data.IMG_ROW, Vbias:Vbias+self.data.IMG_COL, :] = img
                current_att[j] = resize(self.background.copy(), self.data.IMG_SHAPE)
                rimg = self.autoencoder.model.predict(current_att[j].reshape(-1, self.data.IMG_ROW, self.data.IMG_COL, self.data.IMG_CHA))
                # predict
                opred = self.classifier.model.predict(img.reshape(-1, self.data.IMG_ROW, self.data.IMG_COL, self.data.IMG_CHA))
                apred = self.classifier.model.predict(current_att[j].reshape(-1, self.data.IMG_ROW, self.data.IMG_COL, self.data.IMG_CHA))
                rpred = self.classifier.model.predict(rimg.reshape(-1, self.data.IMG_ROW, self.data.IMG_COL, self.data.IMG_CHA))
                # fake status
                if (np.argmax(opred) != np.argmax(apred)) and (np.argmax(opred) != np.argmax(rpred)):
                    current_status.append(True)
                else:
                    current_status.append(False)

            # move current to final
            final_att.extend(current_att)
            final_status.extend(current_status)
        return final_att, final_status















################
# Main
################
if __name__ == '__main__':
    # load data
    mnist = MNIST()
    # init
    attacker = MoveImg(data=mnist)
    # launch
    att, status = attacker.launch()
    # save result
    np.save('results/att.npy', np.array(att))
    np.save('results/status.npy', np.array(status))
        