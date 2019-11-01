import numpy as np
import tensorflow as tf

N_CLASSES = 200
SEMANTIC_SIZE = 28

class Loss():

    def __init__(self, margin_div=None, margin_cct=None):
        self.margin_div = 0.2
        self.margin_cct = 0.8

    # @tf.function
    def loss_CPT(self, map_att, gtmap, batch_size=32):
        diff = tf.math.abs(map_att - gtmap)
        return 2 * tf.nn.l2_loss(diff)/batch_size

    # @tf.function
    def loss_DIV(self, attmap_out):
        print("attention map dim {}".format(attmap_out.shape))
        m_k = attmap_out[0, :, :, :]
        m_i = attmap_out[1, :, :, :]
        n = m_k.shape[0]
        max_mk = tf.math.reduce_max(m_k, axis=[1, 2])
        max_mi = tf.math.reduce_max(m_k, axis=[1, 2])
        max_mk_opp = tf.TensorArray(tf.float32, size=n)
        max_mi_opp = tf.TensorArray(tf.float32, size=n)
        for i in range(n):
            indx_mk = tf.where(m_k[i, :, :] == max_mk[i])
            value_in_mi = tf.gather_nd(m_i[i, :, :], indx_mk) - self.margin_div
            if value_in_mi > 0.0:
                max_mk_opp.write(i, value_in_mi)
            else:
                max_mk_opp.write(i, tf.constant([0.0]))

            indx_mi = tf.where(m_i[i, :, :] == max_mi[i])
            value_in_mk = tf.gather_nd(m_k[i, :, :], indx_mi) - self.margin_div
            if value_in_mi> 0.0:
                max_mi_opp.write(i, value_in_mk)
            else:
                max_mi_opp.write(i, tf.constant([0.0]))
        max_mk_opp = max_mk_opp.stack()
        max_mi_opp = max_mi_opp.stack()
        lossmk = tf.reduce_sum(tf.multiply(tf.expand_dims(max_mk, 1), max_mk_opp))
        lossmi = tf.reduce_sum(tf.multiply(tf.expand_dims(max_mi, 1), max_mi_opp))
        return  lossmk + lossmi

        #m_k_tilt = tf.multiply(max_mk, max_mi - self.margin_div) #tf.math.maximum(m_k - self.margin_div, 0)
        #return tf.tensordot(tf.reshape(m_i,[-1]), tf.reshape(m_k_tilt, [-1]), 1)

    # @tf.function
    def loss_CCT(self, semantic_features, labels, C):
        N = semantic_features.shape[0] * semantic_features.shape[1]
        loss = tf.Variable(0.0)
        norm_C = tf.math.l2_normalize(C, axis=1)
        for f in range(semantic_features.shape[0]):
            sf = semantic_features[f, :, :]
            for s in range(sf.shape[0]):
                diff = tf.math.abs(norm_C - tf.math.l2_normalize(sf[s, :]))
                diff_l2 = tf.reduce_sum(tf.multiply(diff, diff), 1)
                sum_diff_l2 = tf.reduce_sum(diff_l2) - 2*diff_l2[labels[s]] + self.margin_cct
                if sum_diff_l2 > 0.0:
                    loss = loss + sum_diff_l2
        return loss/N

    #@tf.function
    def loss_CLS(self, score):
        exp_score = tf.math.exp(score)
        sum_score = tf.math.reduce_sum(exp_score, [0, 2])
        loss = 0.0
        N = sum_score.shape[0]
        for i in range(N):
            loss += tf.math.log(sum_score[i])
        return loss/N

    def loss_baseline(self, score, labels):
        tf.nn.softmax_cross_entropy_with_logits(score, labels)

    def final_loss(self, attmap_out, crop_out, scores_out, phis_out, y_true, y_pred, n_classes, batch_size, C):
        div = self.loss_DIV(attmap_out)
        cpt = self.loss_CPT(attmap_out, crop_out, batch_size)
        cls = self.loss_CLS(scores_out)
        cct = self.loss_CCT(phis_out, y_true, C)
        return div + cpt + cls + cct



