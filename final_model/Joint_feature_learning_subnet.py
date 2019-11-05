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
        self.W = tf.Variable(initial_value=w_init(shape=(512, semantic_size), dtype='float32'), trainable=True)

    def call(self, thetas, phi):
        n_theta = thetas.shape[0]
        scores_samples = tf.TensorArray(tf.float32, size=n_theta)
        phi_samples = tf.TensorArray(tf.float32, size=n_theta)
        for i in range(n_theta):  # size of theta is (512,)
            theta = thetas[i]
            out = tf.matmul(tf.transpose(tf.reshape(theta, [512, 1])), self.W)  # size of W is (512,28) and shape of out will be (1,28)
            score = tf.matmul(out, phi)  # shape of score is (1,1)
            phi_samples.write(i, out)
            scores_samples.write(i, tf.reshape(score, [1]))  # I am just reshaping it to get rid of one extra redundant dimension
        scores_samples = scores_samples.stack()
        phi_samples = phi_samples.stack()
        return scores_samples, phi_samples


class JFL(tf.keras.Model):
    def __init__(self, n_classes = 200, semantic_size = 28, feature_size = 512):
        super(JFL, self).__init__()
        # mapped feature is a vector of size as semantic feature size (n)
        self.n = semantic_size
        self.embedding = tf.keras.layers.Embedding(self.n,
                                                   n_classes)  # input: class label, embedding_size=semantic_size
        self.W = tf.keras.layers.Dense(self.n, activation="relu", input_shape=(feature_size,))
        self.l2loss = tf.keras.layers.Lambda(
            lambda x: tf.keras.backend.sum(tf.keras.backend.square(x[0] - x[1][:, 0]), 1, keepdims=True))
        self.fc = tf.keras.layers.Dense(n_classes, activation="softmax")

    def call(self, thetas, phi):
        """
        input: theta - visual feature vector
        input: phi - semantic feature vector
        2 'embedding': visual feature(512, 1) ->[W]-> semantic feature(?, 1) and
        semantic feature(?, 1) ->[center]-> classes-semantic embedding (n_classes, ?)
        """
        # trainable class centers, C={c1, c2, ..., c_{n_classes}}
        # initialize: (0, 0.01) Gaussian distribution
        #center = self.embedding(tf.squeeze(phi))  # batch_size, n_classes, semantic_size
        n_thetas = thetas.shape[0]
        scores = tf.TensorArray(tf.float32, size=n_thetas)
        phi_samples = tf.TensorArray(tf.float32, size=n_thetas)
        # need further model integration
        for i in range(n_thetas):
            theta = tf.expand_dims(thetas[i, :], 0)
            # theta = self.reshape(theta)
            out = self.W(theta)  # should have size 1 * n
            phi_samples.write(i, out)
            # compatibility score: s_j^i = theta_i(x)^T W_i phi(y_i)
            scores.write(i, tf.linalg.matmul(out, phi))
            # compute ||~phi_i - ~Ci|| and ||~phi_i - ~Cj||, '~' is normalization
            #l2loss = self.l2loss(
            #    [tf.math.l2_normalize(tf.squeeze(out)), tf.math.l2_normalize(tf.transpose(center, perm=[0, 2, 1]))])
        scores = tf.squeeze(scores.stack())
        phi_samples = tf.squeeze(phi_samples.stack())
        #scores = tf.transpose(tf.squeeze(scores))
        # Normalize the scores???
        # "normalize each descriptor independently, and concatenate them together into
        #  fully-connected fusion layer with softmax function for the final classification. "
        #score = tf.math.reduce_sum(scores, axis=1, keepdims=True)

        return self.normalizer(scores), phi_samples #, l2loss

    def normalizer(self, score):
        mean = tf.math.reduce_mean(score, 1)
        std = tf.math.reduce_std(score, 1)
        new_scores = (score - tf.expand_dims(mean, 1))/tf.expand_dims(std, 1)
        return new_scores


if __name__ == '__main__':
    theta = tf.random.uniform((1, 512), minval=0, maxval=1, dtype=tf.float32)
    phi = tf.random.uniform((1, 200), minval=0, maxval=1, dtype=tf.float32)

    s = Scores()
    s.call(theta, phi)







