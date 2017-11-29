################
# Library
################
import numpy as np
import sys
sys.path.append('/home/guanxiong/Documents/research/Adversarial-Learning/Nov-07/testbed')

from load_data import MNIST
from train_models import Autoencoder
from Clf.train_classifier import Classifier
import tensorflow as tf






################
# Constant
################
B_SIZE  = 128
L_RATE  = 1e-2
INIT_CONST = 1e-3
MAX_CONST = 100
CONST_SEARCH = 10  # how many iteration to find a suitable constant
MAX_ITER = 10000
EARLY_ABORT = True
GEN_NUM = 1  # how many batch of adversarial example to generate












################
# Functions
################
class AntiDef:
    def __init__(self, session, data):
        # get tensorflow session
        self.sess = session

        # load data, autoencoder and classifier
        self.data = data
        self.classifier = Classifier(input_shape=data.IMG_SHAPE, session=self.sess)
        self.classifier.restore('../Clf/models/classifier')
        self.autoencoder = Autoencoder(input_shape=data.IMG_SHAPE, session=self.sess)
        self.autoencoder.restore('models/autoencoder')

        # set parameter
        self.T_SHAPE = (B_SIZE, self.data.IMG_ROW, self.data.IMG_COL, self.data.IMG_CHA)

        # optimize variable
        delta = tf.Variable(np.zeros(self.T_SHAPE), dtype=np.float32)

        # placeholders
        self.timg_PH = tf.placeholder(np.float32, self.T_SHAPE)
        self.const_PH = tf.placeholder(np.float32, (B_SIZE))

        # variables
        self.timg = tf.Variable(np.zeros(self.T_SHAPE), dtype=np.float32)
        self.const = tf.Variable(np.zeros((B_SIZE)), dtype=np.float32)

        # generate original, adversarial and reform img
        self.oimg = self.TanhFunc(self.timg)
        self.aimg = self.TanhFunc(self.timg + delta)
        self.rimg = self.autoencoder.model(self.aimg)

        # evaluate adversarial img
        self.opred = self.classifier.model(self.oimg)
        self.apred = self.classifier.model(self.aimg)
        self.rpred = self.classifier.model(self.rimg)

        # calculate distance
        self.r2odist = tf.reduce_sum(tf.square(self.rimg - self.oimg), [1,2,3])  # reform to original distance
        self.r2adist = tf.reduce_sum(tf.square(self.rimg - self.aimg), [1,2,3]) # reform to adversarial distance
        self.dltdist = tf.reduce_sum(tf.square(delta), [1,2,3]) # delta change distance

        # calculate loss function
        self.loss = tf.reduce_sum(self.dltdist) + tf.reduce_sum(self.const*self.r2adist) - tf.reduce_sum(self.const*self.r2odist)
        #self.loss = tf.reduce_sum(self.dltdist) + tf.reduce_sum(self.const*self.r2adist)

        # setup optimizer and track variables
        start_vars = set(x.name for x in tf.global_variables())
        optimizer = tf.train.AdamOptimizer(L_RATE)
        self.train = optimizer.minimize(self.loss, var_list=[delta])
        end_vars = tf.global_variables()
        new_vars = [x for x in end_vars if x.name not in start_vars]

        # bind input variable and initialize others
        self.input_v = []
        self.input_v.append(self.timg.assign(self.timg_PH))
        self.input_v.append(self.const.assign(self.const_PH))
        self.init = tf.variables_initializer(var_list=[delta]+new_vars)

    def launch(self):
        X = self.data.X_test
        Y = self.data.y_test
        final_att = []
        final_status = []
        for i in range(0, min(X.shape[0], B_SIZE*GEN_NUM), B_SIZE):
            # get current batch data
            batch_X = X[i:i+B_SIZE]
            batch_Y = Y[i:i+B_SIZE]

            # arctan transfer of X
            batch_X = self.ArcTFunc(batch_X)

            # setup const
            batch_const = np.ones(B_SIZE) * INIT_CONST
            const_lb = np.zeros(B_SIZE)
            const_ub = np.ones(B_SIZE) * MAX_CONST

            # score board
            o_mindlt = [1e10] * B_SIZE
            o_bestatt = [np.zeros(batch_X[0].shape)] * B_SIZE
            o_findatt = [False] * B_SIZE

            # search const
            for outer in range(CONST_SEARCH):
                # reset
                self.sess.run(self.init)

                # setup container
                mindlt = [1e10] * B_SIZE

                # setup input
                self.sess.run(self.input_v, {self.timg_PH: batch_X,
                                             self.const_PH: batch_const})

                # inner loop
                pre_l = 1e6
                for inner in range(MAX_ITER):
                    # perform and evaluate attack
                    _, l, dlts, r2as, r2os, aimgs, opreds, apreds, rpreds = self.sess.run([self.train, self.loss, self.dltdist, self.r2adist, 
                                                                                            self.r2odist, self.aimg, self.opred, self.apred, 
                                                                                            self.rpred])

                    # print status
                    if inner%(MAX_ITER//10) == 0:
                        print(inner, l, np.sum(dlts), np.sum(batch_const*r2as), np.sum(batch_const*r2os))
                        if EARLY_ABORT and l > pre_l * .999:
                            break
                        pre_l = l

                    # update inner best attack
                    for e, (dlt,opred,apred,rpred,aimg) in enumerate(zip(dlts, opreds, apreds, rpreds, aimgs)):
                        flg = (np.argmax(opred) != np.argmax(apred)) and (np.argmax(opred) != np.argmax(rpred))
                        if dlt < mindlt[e] and flg:
                            mindlt[e] = dlt
                        if dlt < o_mindlt[e] and flg:
                            o_mindlt[e] = dlt
                            o_bestatt[e] = aimg
                            o_findatt[e] = True

                # update const based on evaluation
                for e in range(B_SIZE):
                    if o_findatt[e]:
                        # find adversarial example, current const is upper bound
                        const_ub[e] = batch_const[e]
                        batch_const[e] = (const_lb[e] + const_ub[e]) / 2.
                    else:
                        # doesn't find adversarial example, current const is lower bound
                        const_lb[e] = batch_const[e]
                        batch_const[e] = (const_lb[e] + const_ub[e]) / 2.

                # print search result
                print('========= The number of attacks find in current batch =========')
                print(len(np.argwhere(np.array(o_findatt) == True).flatten()))
                print('===============================================================')

            # store the best attack of current batch
            final_att.extend(o_bestatt)
            final_status.extend(o_findatt)
        return final_att, final_status

    def TanhFunc(self, in_data):
        res = ((self.data.IMG_MAX-self.data.IMG_MIN)/2.) * tf.tanh(in_data) + ((self.data.IMG_MAX+self.data.IMG_MIN)/2.)
        return res

    def ArcTFunc(self, in_data):
        res = np.arctanh((in_data - ((self.data.IMG_MAX+self.data.IMG_MIN)/2.)) / ((self.data.IMG_MAX-self.data.IMG_MIN)/2.) * 0.999999)
        return res








################
# Main
################
if __name__ == '__main__':
    # load data
    mnist = MNIST()
    with tf.Session() as sess:
        # init
        attacker = AntiDef(session=sess, data=mnist)
        # launch
        att, status = attacker.launch()
        # save result
        np.save('results/att.npy', np.array(att))
        np.save('results/status.npy', np.array(status))
        