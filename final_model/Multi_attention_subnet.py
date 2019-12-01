from tensorflow.keras import layers
import tensorflow as tf
import pathlib
import numpy as np
import matplotlib.pylab as plt
from tensorflow_core.python.keras.models import Sequential
from DataBase import Database

BATCH_SIZE = 32
IMG_SIZE = 448
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
IMG_HEIGHT = 448
IMG_WIDTH = 448
N_CHANNELS = 512

class VGG_feature(layers.Layer):
    '''
    A pre-trained VGG19 network that extracts the features out of images
    Args:
         images = tensor of shape (batch_size, image_size, image_size, 3)
    Returns:
         features = tensor of shape (batch_size, width, height, channels)
    '''
    def __init__(self):
        super(VGG_feature, self).__init__()

    def call(self, x):
        base_model = tf.keras.applications.VGG19(input_shape=IMG_SHAPE,
                                                 include_top=False,
                                                 weights='imagenet')
        feature_batch = base_model(x)
        return feature_batch

class Kmeans(layers.Layer):
    '''
        Clusters up the generated Feature Map into 2 clusters
        Args:
            receives a (batch_size,14,14,512) tensor of batch of features
        Returns:
            two lists, each one contains 32 tensors of shape (512,14,14)
        '''

    def __init__(self, clusters_n=2, iterations=10, total_channels=512):
        super(Kmeans, self).__init__()
        self.clusters_n = clusters_n
        self.iterations = iterations
        self.total_channels = total_channels

    def cluster(self, inputs):
        # center of the clusters
        centroids = tf.Variable(tf.slice(tf.random.shuffle(inputs), [0, 0], [self.clusters_n, -1]),
                                aggregation=tf.VariableAggregation.SUM, trainable=False)
        # cluster assignment: points belong to each cluster
        assignments = tf.Variable(tf.random.uniform((inputs.shape[0],), minval=0, maxval=1, dtype=tf.dtypes.int64),
                                  aggregation=tf.VariableAggregation.SUM, trainable=False)
        t = 0
        while t <= self.iterations:
            # expand
            points_expanded = tf.expand_dims(inputs, 0)
            centroids_expanded = tf.expand_dims(centroids, 1)

            # assign Cluster
            distances = tf.math.reduce_sum(tf.square(tf.subtract(points_expanded, centroids_expanded)), 2)
            assignments.assign(tf.argmin(distances, 0))

            # compute new clusters' values
            means = []
            for c in range(self.clusters_n):
                m = tf.math.reduce_mean(
                    tf.gather(inputs, tf.reshape(tf.where(tf.equal(assignments, c)), [1, -1])), 1)
                means.append(m)
            new_centroids = tf.concat(means, 0)
            centroids.assign(new_centroids)

            t += 1
        return assignments

    def cluster2(self, inputs):
        # center of the clusters
        centroids = tf.Variable(tf.slice(tf.random.shuffle(inputs), [0, 0, 0], [BATCH_SIZE, self.clusters_n, self.clusters_n]),
                                aggregation=tf.VariableAggregation.SUM, trainable=False)
        # cluster assignment: points belong to each cluster
        assignments = tf.Variable(tf.random.uniform((inputs.shape[0:2]), minval=0, maxval=1, dtype=tf.dtypes.int64),
                                  aggregation=tf.VariableAggregation.SUM, trainable=False)
        t = 0
        while t <= self.iterations:
            # expand
            points_expanded = tf.expand_dims(inputs, 1)
            centroids_expanded = tf.expand_dims(centroids, 2)

            # assign Cluster
            distances = tf.math.reduce_sum(tf.square(tf.subtract(points_expanded, centroids_expanded)), 3)
            assignments.assign(tf.argmin(distances, 1))

            # compute new clusters' values
            new_centroids = []
            for b in range(BATCH_SIZE):
                assig0 = tf.cast(assignments[b, :], tf.dtypes.bool)
                m0 = tf.expand_dims(tf.math.reduce_mean(
                    tf.gather_nd(inputs[b, :, :], tf.where(assig0)), 0), 0)  # tf.reshape( , [1, -1])
                assig1 = tf.where(assig0, 0, 1)
                m1 = tf.expand_dims(tf.math.reduce_mean(
                    tf.gather_nd(inputs[b, :, :], tf.where(assig1)), 0), 0) # tf.reshape( , [1, -1])
                batch_centroids = tf.expand_dims(tf.concat([m0, m1], 0), 0)
                new_centroids.append(batch_centroids)
            centroids.assign(tf.concat(new_centroids, 0))
            t += 1

        return assignments

    def get_max_pixels(self, batch):
        batch_t = tf.transpose(batch, [0, 3, 1, 2])

        max1 = tf.reduce_max(batch_t, axis=2)
        arg1 = tf.argmax(max1, axis=2)
        arg1 = tf.expand_dims(arg1, 2)

        max0 = tf.reduce_max(batch_t, axis=3)
        arg0 = tf.argmax(max0, axis=2)
        arg0 = tf.expand_dims(arg0, 2)

        max_pixels = tf.concat([arg1, arg0], axis=2)
        return max_pixels

    def create_condition(self, assignments, cond_value):
        # get right dimension for assignments
        total_assig0 = []
        total_assig1 = []
        l0 = tf.zeros((1, 14, 14))
        l1 = tf.ones((1, 14, 14))
        for b in range(BATCH_SIZE):
            assig0 = []
            assig1 = []

            def f0():
                return l0, l1

            def f1():
                return l1, l0
            for c in range(N_CHANNELS):
                value = tf.gather_nd(assignments, [b, c])
                v1, v0 = tf.case([(tf.greater(value, 0), f1)], f0)
                assig0.append(v0)
                assig1.append(v1)
                # if assignments[b, c]:
                #     assig.append(l1)
                # else:
                #     assig.append(l0)
            total_assig0.append(tf.expand_dims(tf.concat(assig0, 0), 0))
            total_assig1.append(tf.expand_dims(tf.concat(assig1, 0), 0))
        condition0 = tf.cast(tf.concat(total_assig0, 0), tf.dtypes.bool)
        condition1 = tf.cast(tf.concat(total_assig1, 0), tf.dtypes.bool)
        return condition1, condition0

    def call(self, feature_batch):
        max_points = self.get_max_pixels(feature_batch)

        assignments = self.cluster2(max_points) #tf.cast(, tf.dtypes.bool)
        batch_images = tf.transpose(feature_batch, [0, 3, 1, 2])
        image0 = tf.zeros(batch_images.shape)

        condition1, condition0 = self.create_condition(assignments, 1)
        cluster1 = tf.where(condition1, batch_images, image0)
        #condition0 = self.create_condition(assignments, 0) #tf.where(assignments, 0, 1))
        cluster0 = tf.where(condition0, batch_images, image0)

        C0 = tf.transpose(cluster0, [0, 2, 3, 1])
        C1 = tf.transpose(cluster1, [0, 2, 3, 1])
        return C0, C1

        # for b in range(BATCH_SIZE):
        #     cluster0 = []
        #     cluster1 = []
        #     assigned0 = tf.where(tf.equal(assignments[b, :], 0))
        #     for e in range(N_CHANNELS):
        #         image = tf.expand_dims(tf.gather_nd(batch_images, [b, e]), 0)
        #         if e in assigned0:
        #             cluster0.append(image)
        #             cluster1.append(image0)
        #         else:
        #             cluster0.append(image0)
        #             cluster1.append(image)
        #     batch_cluster0.append(tf.concat(cluster0, 0))
        #     batch_cluster1.append(tf.concat(cluster1, 0))
        # C0 = tf.concat(batch_cluster0, 0)
        # C1 = tf.concat(batch_cluster1, 0)

# this is a new version of Average Pooling
class Average_Pooling(layers.Layer):
    '''
    Args:
         tesnor (32,14,14,512)
    Returns:
        tensor (32,512)

    if you think its wrong,we canuse a layer from keras instead:
    https://www.tensorflow.org/api_docs/python/tf/keras/layers/AveragePooling2D

    '''

    def __init__(self):
        super( Average_Pooling, self).__init__()

    def call(self, cluster):
        out = tf.math.reduce_mean(cluster, axis=(1, 2))
        return out

#if you think its wrong,we canuse a layer from keras instead:
#https://www.tensorflow.org/api_docs/python/tf/keras/layers/AveragePooling2D


"""
class Average_Pooling(layers.Layer):
    '''
    Args:
        a list of  32(=batch_size) tensors of size (512,14,14)
    Returns:
        a list of 32(=batch_size) tensors of size (512,)
    '''

    def __init__(self):
        super(Average_Pooling, self).__init__()

    def call(self, cluster):
        n_cluster = cluster.shape[0]
        p_batch = tf.TensorArray(tf.float32, size=n_cluster)
        for i in range(n_cluster):
            b = cluster[i, :, :, :]
            H, W = b.shape[0], b.shape[1]
            p = tf.math.reduce_sum(b, axis=(0, 1))/(H*W)
            p_batch.write(i, p)
        p_batch = p_batch.stack()
        print("p_batch shape {}".format(p_batch.shape))
        return p_batch

class Average_Pooling_basemodel(layers.Layer):
    '''
    Args:
        a list of  32(=batch_size) tensors of size (512,14,14)
        a TENSOR of size (batch,14,14,512)
    Returns:
        a list of 32(=batch_size) tensors of size (512,)
    '''

    def __init__(self):
        super(Average_Pooling_basemodel, self).__init__()

    def call(self, cluster):
        cluster = tf.unstack(cluster, axis=0)
        p_batch = []
        for b in cluster:
            H, W = b.shape[0], b.shape[1]
            p = tf.math.reduce_sum(b, axis=(0, 1))/(H*W)
            p_batch.append(p)
        return p_batch
        
        
"""

class Fc(layers.Layer):
    '''
    As a part of the Multi-Attention subnet it will return the weight vector
        Args:
            a list of  32(=batch_size) tensors of size (512,)
        Returns:
            A tensor of size (32,512)=(batch_size , vector of length equal to number of channels)
        '''

    def __init__(self, input_shape):
        super(Fc, self).__init__()
        self.initializer = tf.keras.initializers.glorot_normal()
        self.fc1 = tf.keras.layers.Dense(input_shape, input_shape=(input_shape,), activation="relu", kernel_initializer=self.initializer)
        self.fc2 = tf.keras.layers.Dense(input_shape, activation="sigmoid", kernel_initializer=self.initializer)
        self.bn = tf.keras.layers.BatchNormalization()
        self.a_batch = tf.TensorArray(tf.float32, size=BATCH_SIZE)

    def call(self, p_batch):
        n_p_batch = p_batch.shape[0]
        #a_batch = tf.TensorArray(tf.float32, size=n_p_batch)
        for i in range(n_p_batch):
            p = p_batch[i]
            p = tf.expand_dims(p, 0)
            out = self.fc1(p)
            out = self.fc2(out)
            a = self.bn(out)
            self.a_batch.write(i, a)
        a_batch = tf.squeeze(self.a_batch.stack())
        return a_batch


class WeightedSum(layers.Layer):
    """
    Generates the final attention maps focused on certain parts of the image.
    Args:
        feature map (32,14,14,512)  and a tensor (weight vector) of shape(32,512)
    Returns:
        a tensor of size (32,14,14). We sum along axis=3 (that is 512) using the weight vector.
    """

    def __init__(self):
        super(WeightedSum, self).__init__()

    def call(self, feature_batch, weight_vector):

        # expand dims twice to make the weight vector to be of shape ([32, 1, 1, 512])
        weight_vector = tf.expand_dims(weight_vector, axis=1)
        weight_vector = tf.expand_dims(weight_vector, axis=2)

        product = tf.math.multiply(feature_batch, weight_vector) # product shape is (32,14,14,512)
        summed_product = tf.math.reduce_sum(product, axis=3)  # summed product shape is (32,14,14)
        denominator = tf.math.reduce_sum(weight_vector, axis=3) # will be dividing by this (shape=(32,1,1))

        return summed_product/denominator

'''
class WeightedSum(layers.Layer):
    """
    Generates the final attention maps focused on certain parts of the image.
    Args:
        a tensor of shape(32,512)
    Returns:
        a tensor of size (32,14,14)
    """

    def __init__(self):
        super(WeightedSum, self).__init__()
        self.attention_maps_batch = tf.TensorArray(tf.float32, size=BATCH_SIZE)

    def call(self, batch, a_batch):
        n_batch = batch.shape[0]
        #attention_maps_batch = tf.TensorArray(tf.float32, size=n_batch)
        for b in range(n_batch):
            image = tf.zeros((batch.shape[1], batch.shape[2]))
            for c in range(batch.shape[-1]):
                w_image = tf.multiply(batch[b, :, :, c], a_batch[b, c])
                image += w_image
            image /= tf.math.reduce_max(image)
            self.attention_maps_batch.write(b, image)
        attention_maps_batch = self.attention_maps_batch.stack()
        return attention_maps_batch
'''

import pickle

def saveVariables(self, variables): #where 'variables' is a list of variables
    with open("nameOfYourFile.txt", 'wb+') as file:
       pickle.dump(variables, file)

def retrieveVariables(self, filename):
    variables = []
    with open(str(filename), 'rb') as file:
        variables = pickle.load(file)
    return variables

if __name__ == '__main__':
    database = Database()
    image_batch, label_batch = database.call()

    vgg = VGG_feature()
    feature_batch = vgg.call(image_batch)

    km = Kmeans()
    batch_cluster0, batch_cluster1 = km.call(feature_batch)

    average_pooling_0 = Average_Pooling()
    pb0 = average_pooling_0.call(batch_cluster0)
    average_pooling_1 = Average_Pooling()
    pb1 = average_pooling_1.call(batch_cluster1)

    fc_0 = Fc(512)
    a_batch0 = fc_0.call(pb0)
    fc_1 = Fc(512)
    a_batch1 = fc_1.call(pb1)

    weighted_sum0 = WeightedSum()
    im0 = weighted_sum0.call(feature_batch, a_batch0)
    weighted_sum1 = WeightedSum()
    im1 = weighted_sum1.call(feature_batch, a_batch1)

    print("end kmeans")
