from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Activation, GlobalAveragePooling2D
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
import matplotlib.pyplot as plt
from tensorflow.keras import layers
import numpy as np


IMG_SIZE = 448
n_classes  = 200
semantic_size = 28

class Scores(layers.Layer):
    """
      Args:
          thetas: a list containing 32 visual feature vectors of size ()
          phi: semantic feature vector
      Returns:
          call function returns a list of compatibility score- tensors
    """
    def __init__(self):
        super(Scores, self).__init__()
        w_init = tf.random_normal_initializer()
        self.W = tf.Variable(initial_value=w_init(shape=(512,semantic_size), dtype='float32'), trainable=True)

    def call(self, thetas, phi):
        scores=[]
        for theta in thetas: #size of theta is (512,)
            out = tf.matmul(tf.transpose(tf.reshape(theta,[512,1])), self.W) #size of W is (512,28) and shape of out will be (1,28)
            score = tf.matmul(out, phi) #shape of score is (1,1)
            scores.append(tf.reshape(score,[1])) # I am just reshaping it to get rid of one extra redundant dimension
        scores = tf.stack(scores)
        return scores, out













