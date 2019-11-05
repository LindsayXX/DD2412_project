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

    def call(self, global_scores, local_scores0, local_scores1):
        sum_scores = tf.math.reduce_sum(scores, [0])
        y_pred = tf.math.argmax(sum_scores, 1)
        return y_pred
