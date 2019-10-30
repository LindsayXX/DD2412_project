import numpy as np
import tensorflow as tf

class Loss():

    def __init__(self, margin_div=None, margin_cct=None):
        self.margin_div = margin_div
        self.margin_cct = margin_cct

    # @tf.function
    def loss_CPT(self, map, gtmap, batch_size=32):
        diff = tf.math.abs(map, gtmap)
        return tf.nn.l2_loss(diff)/batch_size

    # @tf.function
    def loss_DIV(self, m_i, m_k):
        m_k_tilt = tf.math.maximum(m_k - self.margin, 0)
        return tf.tensordot(tf.reshape(m_i,[-1]), tf.reshape(m_k_tilt, [-1]), 1)

    # @tf.function
    def loss_CCT(self, y_true, y_pred, n_classes):
        # triplet center loss
        # Original implementation: https://github.com/popcornell/keras-triplet-center-loss/blob/master/triplet.py
        # y_true -- semantic class center
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

        masked = y_pred_r[:, :, 0] * tf.cast(mask,
                                             tf.float32)  # + (mask2 * tf.cast(tf.logical_not(mask), tf.float32))*tf.constant(float(2**10))
        masked = tf.where(tf.equal(masked, 0.0), np.inf * tf.ones_like(masked), masked)

        minimum = tf.math.reduce_min(masked, axis=1)
        loss = tf.keras.backend.max(y_pred - minimum + self.margin_cct, 0)

        return loss

    #@tf.function
    def loss_CLS(self, score):
        exp_score = tf.math.exp(score)
        sum_score = tf.math.reduce_sum(exp_score)
        loss = 0.0
        N = score.size[0]
        for i in range(N):
            loss += tf.math.log(sum_score[i])
        return loss/N

    def loss_baseline(self, score, labels):
        tf.nn.softmax_cross_entropy_with_logits(score, labels)



