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
class JFL(tf.keras.Model):
    def __init__(self, n_classes = 150, semantic_size = 312, feature_size = 512):
        super(JFL, self).__init__()
        # mapped feature is a vector of size as semantic feature size (n)
        self.n = semantic_size
        self.W = tf.keras.layers.Dense(self.n, activation="relu", input_shape=(feature_size,))
        self.l2loss = tf.keras.layers.Lambda(
            lambda x: tf.keras.backend.sum(tf.keras.backend.square(x[0] - x[1][:, 0]), 1, keepdims=True))
        self.fc = tf.keras.layers.Dense(n_classes, activation="softmax")
        # trainable class centers, C={c1, c2, ..., c_{n_classes}}
        c_init = tf.random_normal_initializer(0, 0.01)
        self.C = tf.Variable(initial_value=c_init(shape=(semantic_size, n_class), dtype='float32'), trainable=True,name="C")

    def call(self, theta, Phi):
        """
        input: theta - visual feature of size: batch_size * 512
        input: phi - semantic feature matrix: 312 * 150(n_classes)
        2 'embedding': visual feature(512, 1) ->[W]-> semantic feature(312, 1) and
        semantic feature(312, 1) ->[center]-> classes-semantic embedding (n_classes, ?)
        """
        out = self.W(theta)  # should have size 1 * n_semantic
        # compatibility score: s_j^i = theta_i(x)^T W_i phi(y_i)
        scores = tf.linalg.matmul(out, Phi) # should have size 1 * n_classes
        # compute ||~phi_i - ~Ci|| and ||~phi_i - ~Cj||, '~' is normalization
        #l2loss = self.l2loss(
        #    [tf.math.l2_normalize(tf.squeeze(out)), tf.math.l2_normalize(tf.transpose(center, perm=[0, 2, 1]))])
        # Normalize the scores?
        # "normalize each descriptor independently, and concatenate them together into
        #  fully-connected fusion layer with softmax function for the final classification. "
        score = tf.math.reduce_sum(scores, axis=1, keepdims=True)

        return score, out #, l2loss

    def normalizer(self, score):
        mean = tf.math.reduce_mean(score, 1)
        std = tf.math.reduce_std(score, 1)
        new_scores = (score - tf.expand_dims(mean, 1))/tf.expand_dims(std, 1)
        return new_scores

# @tf.function
def CCT_loss(y_true, y_pred, n_classes, margin=0.8):
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
    loss = tf.keras.backend.max(y_pred - minimum + margin, 0)

    return loss


# @tf.funcion
def CLS(score, label):
    return tf.nn.softmax_cross_entropy_with_logits(score, label)


if __name__ == '__main__':
    # test
    '''
    base_model = tf.keras.applications.VGG19(input_shape=IMG_SHAPE,
                                             include_top=False,
                                             weights='imagenet')
    model = tf.keras.Model(inputs=base_model.input, outputs=base_model.get_layer('block3_pool').output)
    feature_batch = base_model(image_batch)
    
    '''
    batch_size = 5
    n_class = 100
    semantic_size = 28
    feature_size = 512
    sample_feature_1 = tf.random.normal([batch_size, 1, 512])  # global feature
    sample_feature_2 = tf.random.normal([batch_size, 1, 512])  # local feature1(head)
    sample_feature_3 = tf.random.normal([batch_size, 1, 512])  # local feature2(tail)
    sample_feature = tf.convert_to_tensor(np.load("../final_model/theta_global.npy")) #
    sample_semantic = tf.convert_to_tensor(np.load("../final_model/phi.npy")) #tf.ones([batch_size, semantic_size, 1])
    sample_labels = tf.zeros([batch_size, 1, n_class])
    joint_net = JFL(n_class, semantic_size, feature_size)
    #sample_feature = [sample_feature_1, sample_feature_2, sample_feature_3]
    y_pred = joint_net.call(sample_feature, sample_semantic)
