import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D, Activation, GlobalAveragePooling2D
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
import matplotlib.pyplot as plt
import numpy as np

IMG_SIZE = 448
'''Multi-Attention Subnet'''


'''Region Cropping Subnet'''


'''Joint Feature Learning Subnet'''
class JFL(tf.keras.models):
    def __init__(self, n_classes, semantic_size):
        # mapped feature is a vector of size as semantic feature size (n)
        self.n = semantic_size
        #self.reshape = tf.keras.layers.Reshape(-1, 1, -1)
        self.W = tf.keras.layers.Dense(self.n, activation="relu")#, input_shape=(map_size * map_size))
        self.embedding = tf.keras.layers.Embedding(n_classes, self.n)
        self.l2loss = tf.keras.layers.Lambda(lambda x: tf.keras.backend.sum(tf.keras.backend.square(x[0] - x[1][:, 0]), 1, keepdims=True))
        #

    def call(self, theta, phi):
        # input: theta - visual feature vector
        # input: phi - semantic feature vector
        out = self.reshape(theta)
        out = self.W(out) # should have size 1 * n
        # compatibility score: s_j^i = theta_i(x)^T W_i phi(y_i)
        score = tf.linalg.matmul(out, phi)
        # trainable class centers
        center = self.embedding(phi)
        l2loss = self.l2loss([tf.math.l2_normalize(out), tf.math.l2_normalize(center)])

        return score, l2loss


#@tf.function
def CLS_loss(score1, score2, score3, labels):
    # embedding softmax loss - softmax with cross entropy loss
    # use late fusion strategy -- combine the outputs of the 3 networks
    # through their last fully-connected layer by score summing
    # Assume the scores have the same labels
    scores = tf.math.add_n([score1, score2, score3])
    loss = tf.nn.softmax_cross_entropy_with_logits(labels, scores)

    return loss


#@tf.function
def CCT_loss(y_true, y_pred, n_classes, margin=0.8):
    # triplet center loss
    # modified from https://github.com/popcornell/keras-triplet-center-loss/blob/master/triplet.py
    # y_true -- semantic feature
    # y_pred -- embedding feature
    print('y_pred.shape = ', y_pred.shape)
    print('total_lengh =', y_pred.shape.as_list()[-1])

    # repeat y_true for n_classes and == np.arange(n_classes)
    # repeat also y_pred and apply mask
    # obtain min for each column min vector for each class

    classes = tf.range(0, n_classes, dtype=tf.float32)
    y_pred_r = tf.reshape(y_pred, (tf.shape(y_pred)[0], 1))
    y_pred_r = tf.keras.backend.repeat(y_pred_r, n_classes)

    y_true_r = tf.reshape(y_true, (tf.shape(y_true)[0], 1))
    y_true_r = tf.keras.backend.repeat(y_true_r, n_classes)

    mask = tf.equal(y_true_r[:, :, 0], classes)
    # mask2 = tf.ones((tf.shape(y_true_r)[0], tf.shape(y_true_r)[1]))  # todo inf
    # use tf.where(tf.equal(masked, 0.0), np.inf*tf.ones_like(masked), masked)

    masked = y_pred_r[:, :, 0] * tf.cast(mask, tf.float32)  # + (mask2 * tf.cast(tf.logical_not(mask), tf.float32))*tf.constant(float(2**10))
    masked = tf.where(tf.equal(masked, 0.0), np.inf * tf.ones_like(masked), masked)

    minimums = tf.math.reduce_min(masked, axis=1)
    loss = tf.keras.backend.max(y_pred - minimums + margin, 0)
    '''
    anchor = y_pred[0]
    pos = y_pred[1]
    neg = y_pred[2]
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, pos)), axis=-1)
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, neg)), axis=-1)
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), self.margin)
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))
    '''
    return loss


if __name__ == '__main__':
    # test
    '''
    base_model = tf.keras.applications.VGG19(input_shape=IMG_SHAPE,
                                             include_top=False,
                                             weights='imagenet')
    model = tf.keras.Model(inputs=base_model.input, outputs=base_model.get_layer('block3_pool').output)
    feature_batch = base_model(image_batch)
    
    '''
    batch_size = 3
    sample_feature = tf.random.normal([batch_size, 14, 14, 1])
    sample_semantic = tf.ones([batch_size, 20])
    joint_net = JFL(n_classes=100, semantic_size=20)
    score, l2loss = joint_net.call(sample_feature, sample_semantic)