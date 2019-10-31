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
    def __init__(self):
        super(self).__init__()

    def call(self, scores):
        sum_scores = tf.math.reduce_sum(scores, [0])
