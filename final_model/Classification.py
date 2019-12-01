from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras import layers

class Classifier(layers.Layer):
    """
      Args:
          thetas: a list containing 32 visual feature vectors of size ()
          phi: semantic feature vector
      Returns:
          call function returns a list of compatibility score- tensors
    """
    def __init__(self, n_classes):
        super(Classifier, self).__init__()
        self.n_classes = n_classes

    def call(self, scores):
        y_pred = tf.math.argmax(scores, 1)
        return y_pred


class Classifier_Unseen(layers.Layer):
    """
      Args:
          thetas: a list containing 32 visual feature vectors of size ()
          phi: semantic feature vector
      Returns:
          call function returns a list of compatibility score- tensors
    """
    def __init__(self, W, C):
        super(Classifier_Unseen, self).__init__()
        self.W = W
        self.PHI_cct = tf.tensordot(W, C)
        self.beta = 1.0

    def call(self, phi, scores):
        b_phi = tf.tensordot(phi, self.PHI_cct)
        sum_s_phi = scores + tf.multiply(self.beta, b_phi)
        argmax = tf.argmax(sum_s_phi)

        return tf.gather_nd(self.unseen_classes, argmax)
