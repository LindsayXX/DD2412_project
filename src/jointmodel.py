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
    def __init__(self, n_classes, semantic_size):
        super(JFL, self).__init__()
        # mapped feature is a vector of size as semantic feature size (n)
        self.n = semantic_size
        self.embedding = tf.keras.layers.Embedding(self.n, n_classes)#input: class label, embedding_size=semantic_size
        self.W = tf.keras.layers.Dense(self.n, activation="relu")# input_shape=(feature_size))
        self.l2loss = tf.keras.layers.Lambda(lambda x: tf.keras.backend.sum(tf.keras.backend.square(x[0] - x[1][:, 0]), 1, keepdims=True))
        self.fc = tf.keras.layers.Dense(n_classes)


    def call(self, thetas, phi):
        """
        input: theta - visual feature vector
        input: phi - semantic feature vector
        2 'embedding': visual feature(512, 1) ->[W]-> semantic feature(?, 1) and
        semantic feature(?, 1) ->[center]-> classes-semantic embedding (n_classes, ?)
        """
        # trainable class centers, C={c1, c2, ..., c_{n_classes}}
        # initialize: (0, 0.01) Gaussian distribution
        center = self.embedding(tf.squeeze(phi))# batch_size, n_classes, semantic_size
        scores = []
        # need further model integration
        for i in range(len(thetas)):
            theta = thetas[i]
            #theta = self.reshape(theta)
            out = self.W(theta) # should have size 1 * n
            # compatibility score: s_j^i = theta_i(x)^T W_i phi(y_i)
            scores.append(tf.linalg.matmul(out, phi))
            # compute ||~phi_i - ~Ci|| and ||~phi_i - ~Cj||, '~' is normalization
            l2loss = self.l2loss([tf.math.l2_normalize(tf.squeeze(out)), tf.math.l2_normalize(tf.transpose(center, perm=[0, 2, 1]))])
        scores = tf.stack(scores)
        scores = tf.transpose(tf.squeeze(scores))
        # Normalize the scores???
        # "normalize each descriptor independently, and concatenate them together into
        #  fully-connected fusion layer with softmax function for the final classification. "
        score = tf.math.reduce_sum(scores, axis=1, keepdims=True)
        softmax = self.fc(out)

        return softmax #, l2loss


#@tf.function
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

    masked = y_pred_r[:, :, 0] * tf.cast(mask, tf.float32)  # + (mask2 * tf.cast(tf.logical_not(mask), tf.float32))*tf.constant(float(2**10))
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
    semantic_size = 20
    sample_feature_1 = tf.random.normal([batch_size, 1, 512])# global feature
    sample_feature_2 = tf.random.normal([batch_size, 1, 512])# local feature1(head)
    sample_feature_3 = tf.random.normal([batch_size, 1, 512])# local feature2(tail)
    sample_semantic = tf.ones([batch_size, semantic_size, 1])
    sample_labels = tf.zeros([batch_size, 1, n_class])
    joint_net = JFL(n_class, semantic_size)
    sample_feature = [sample_feature_1, sample_feature_2, sample_feature_3]
    y_pred = joint_net.call(sample_feature, sample_semantic)

