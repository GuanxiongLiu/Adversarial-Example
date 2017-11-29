###########################
# Library
###########################
import os
import numpy as np

from keras.datasets import mnist
from keras.models import load_model

import tensorflow as tf










###########################
# Pre-defined Functions
###########################
# about image
IMG_ROW = 28
IMG_COL = 28
IMG_CHA = 1
IMG_CLA = 10
IMG_MIN = 0
IMG_MAX = 1



# about optimization 
B_SIZE  = 128
T_SHAPE = (B_SIZE, IMG_ROW, IMG_COL, IMG_CHA)
L_RATE  = 1e-2
INIT_CONST = 1e-3
CONST_SEARCH = 10
MAX_ITER = 1e4



# functions
def TanhFunc(in_data):
    res = ((IMG_MAX-IMG_MIN)/2.) * tf.tanh(in_data) + ((IMG_MAX+IMG_MIN)/2.)
    return res

def ArcTFunc(in_data):
    res = np.arctanh((in_data - ((IMG_MAX+IMG_MIN)/2.)) / ((IMG_MAX-IMG_MIN)/2.) * 0.999999)
    return res








###########################
# Main Function
###########################
if __name__ == '__main__':
    print('========= Loading Dataset =========')
    (X_train, Y_train), (_, _) = mnist.load_data()
    X = (X_train.astype('float32') / 255.).reshape(-1, IMG_ROW, IMG_COL, IMG_CHA)
    Y = Y_train



    print('========= Loading Autoencoder =========')
    autoencoder = load_model('./autoencoder.h5')



    print('========= Loading Classifier =========')
    classifier = load_model('./classifier.h5')



    print('========= Building TensorFlow Model =========')
    # get tensorflow session
    sess = tf.Session()

    # optimize variable
    delta = tf.Variable(np.zeros(T_SHAPE), dtype=np.float32)

    # placeholders
    timg_PH = tf.placeholder(np.float32, T_SHAPE)
    const_PH = tf.placeholder(np.float32, (B_SIZE))

    # variables
    timg = tf.Variable(np.zeros(T_SHAPE), dtype=np.float32)
    const = tf.Variable(np.zeros((B_SIZE)), dtype=np.float32)

    # generate adversarial img
    aimg = TanhFunc(timg+delta)

    # reform by autoencoder
    rimg = autoencoder(aimg)

    # calculate distance
    oimg = TanhFunc(timg)
    r2odist = tf.reduce_sum(tf.square(rimg - oimg), [1,2,3])  # reform to original distance
    r2adist = tf.reduce_sum(tf.square(rimg - aimg), [1,2,3]) # reform to adversarial distance
    dltdist = tf.reduce_sum(tf.square(delta), [1,2,3]) # delta change distance

    # calculate loss function
    loss = tf.reduce_sum(dltdist) + tf.reduce_sum(const*r2adist) - tf.reduce_sum(const*r2odist)

    # setup optimizer and track variables
    start_vars = set(x.name for x in tf.global_variables())
    optimizer = tf.train.AdamOptimizer(L_RATE)
    train = optimizer.minimize(loss, var_list=[delta])
    end_vars = tf.global_variables()
    new_vars = [x for x in end_vars if x.name not in start_vars]

    # bind input variable and initialize others
    input_v = []
    input_v.append(timg.assign(timg_PH))
    input_v.append(const.assign(const_PH))
    init = tf.variables_initializer(var_list=[delta]+new_vars)



    print('========= Finding Adversarial Examples =========')
    final_att = []
    final_status = []
    for i in range(0, X.shape[0], B_SIZE):
        # get current batch data
        batch_X = X[i:i+B_SIZE]
        batch_Y = Y[i:i+B_SIZE]

        # arctan transfer of X
        batch_X = ArcTFunc(batch_X)

        # setup const
        batch_const = np.ones(B_SIZE) * INIT_CONST
        const_lb = np.zeros(B_SIZE)
        const_ub = np.ones(B_SIZE) * 1e10

        # score board
        o_mindlt = [1e10] * B_SIZE
        o_bestatt = [np.zeros(batch_X[0].shape)] * B_SIZE
        o_findatt = [False] * B_SIZE

        # search const
        for outer in range(CONST_SEARCH):
            # reset
            sess.run(init)

            # setup container
            mindlt = [1e10] * B_SIZE

            # setup input
            sess.run(input_v, {timg_PH: batch_X,
                               const_PH: batch_const})

            # inner loop
            pre_l = 1e6
            for inner in range(int(MAX_ITER)):
                # perform and evaluate attack
                _, l, dlts, r2as, r2os, nimg = sess.run([train, loss, dltdist, 
                                                      r2adist, r2odist, aimg])
                preds = classifier(nimg)

                # print status
                if inner%(MAX_ITER//10) == 0:
                    print(inner, sess.run((loss, dltdist, r2adist, r2odist)))
                    if l > pre_l * .9999:
                        break
                    pre_l = l

                # update inner best attack
                for e, (dlt,pred,img) in enumerate(zip(dlts, preds, nimg)):
                    if dlt < mindlt[e] and np.argmax(pred) != batch_Y[e]:
                        mindlt[e] = dlt
                    if dlt < o_mindlt[e] and np.argmax(pred) != batch_Y[e]:
                        o_mindlt[e] = dlt
                        o_bestatt[e] = img
                        o_findatt[e] = True

            # update const based on evaluation
            for e in range(B_SIZE):
                if o_findatt[e]:
                    # find adversarial example, current const is upper bound
                    const_ub[e] = min(const_ub[e], batch_const[e])
                    batch_const[e] = (const_lb[e] + const_ub[e]) / 2.
                else:
                    # doesn't find adversarial example, current const is lower bound
                    const_lb[e] = max(const_lb[e], batch_const[e])
                    batch_const[e] = (const_lb[e] + const_ub[e]) / 2.

        # store the best attack of current batch
        final_att.extend(o_bestatt)
        final_status.extend(o_findatt)



    print('========= Saving Results =========')
    # store the attacking result
    np.save('adversarial_examples.npy', np.arrary(final_att))
    np.save('searching_result.npy', np.array(final_status))





        



















