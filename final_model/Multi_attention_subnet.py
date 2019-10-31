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
        centroids = tf.Variable(tf.slice(tf.random.shuffle(inputs), [0, 0], [self.clusters_n, -1]))
        # cluster assignment: points belong to each cluster
        assignments = tf.Variable(tf.random.uniform((inputs.shape[0],), minval=0, maxval=1, dtype=tf.dtypes.int64))
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

    def get_max_pixels(self, batch):
        max_batch = tf.TensorArray(tf.int64, size=batch.shape[0])
        for i in range(batch.shape[0]):
            image_batch = batch[i, :, :, :]
            max_pix = tf.TensorArray(tf.int64, size=batch.shape[-1])
            for channel in range(image_batch.shape[2]):
                image = image_batch[:, :, channel]
                maximum = tf.math.reduce_max(image)
                x = tf.where(image == maximum)
                if x.shape[0] == 1:
                    max_pix.write(channel, x) #max_pix.append(x)
                else:
                    # continue
                    choice = tf.dtypes.cast(tf.random.uniform((1, 1), 0, x.shape[0]), tf.int64)[0][0]
                    max_pix.write(channel, tf.expand_dims(x[choice], 0)) #.append(tf.expand_dims(x[choice], 0))
            max_pix = max_pix.stack()
            max_batch.write(i, max_pix)
        max_batch = tf.reshape(max_batch.stack(), [batch.shape[0], batch.shape[-1], 2])
        return max_batch

    def call(self,feature_batch):
        max_points = self.get_max_pixels(feature_batch)
        n_max_points = max_points.shape[0]
        batch_cluster0 = tf.TensorArray(tf.float32, size=n_max_points)
        batch_cluster1 = tf.TensorArray(tf.float32, size=n_max_points)
        for m in range(n_max_points):
            assign = self.cluster(max_points[m, :, :])
            n_assign = assign.shape[0]
            cluster0 = tf.TensorArray(tf.float32, size=n_assign)
            cluster1 = tf.TensorArray(tf.float32, size=n_assign)
            for v in range(n_assign):
                image = feature_batch[m, :, :, v]
                if assign[v] == 0:
                    #cluster0.append(image)
                    cluster0.write(v, image)
                    image = tf.zeros((14, 14))
                    cluster1.write(v, image)
                else:
                    cluster1.write(v, image)
                    image = tf.zeros((14, 14))
                    cluster0.write(v, image)

            cluster0 = cluster0.stack()
            cluster1 = cluster1.stack()
            batch_cluster0.write(m, cluster0)
            batch_cluster1.write(m, cluster1)
        batch_cluster0 = tf.transpose(batch_cluster0.stack(), perm=[0, 2, 3, 1])
        batch_cluster1 = tf.transpose(batch_cluster1.stack(), perm=[0, 2, 3, 1])
        return batch_cluster0, batch_cluster1


class Average_Pooling(layers.Layer):
    '''
    Args:
        a list of  32(=batch_size) tensors of size (512,14,14)
    Returns:
        a list of 32(=batch_size) tensors of size (512,)
    '''

    def __init__(self):
        super( Average_Pooling, self).__init__()

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
        super( Average_Pooling_basemodel, self).__init__()

    def call(self, cluster):
        cluster = tf.unstack(cluster, axis=0)
        p_batch = []
        for b in cluster:
            H, W = b.shape[0], b.shape[1]
            p = tf.math.reduce_sum(b, axis=(0, 1))/(H*W)
            p_batch.append(p)
        return p_batch

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

    def call(self, p_batch):
        n_p_batch = p_batch.shape[0]
        a_batch = tf.TensorArray(tf.float32, size=n_p_batch)
        for i in range(n_p_batch):
            p = p_batch[i]
            p = tf.expand_dims(p, 0)
            out = self.fc1(p)
            out = self.fc2(out)
            a = self.bn(out)
            a_batch.write(i, a)
        a_batch = tf.squeeze(a_batch.stack())
        return a_batch


class WeightedSum(layers.Layer):
    """
    Generates the final attention maps focused on certain parts of the image.

    Args:
        a tensor of shape()32,512
    Returns:
        a tensor of size (32,14,14)
    """

    def __init__(self):
        super(WeightedSum, self).__init__()

    def call(self, batch, a_batch):
        n_batch = batch.shape[0]
        attention_maps_batch = tf.TensorArray(tf.float32, size=n_batch )
        for b in range(n_batch):
            image = tf.zeros((batch.shape[1], batch.shape[2]))
            for c in range(batch.shape[-1]):
                w_image = tf.multiply(batch[b, :, :, c], a_batch[b, c])
                image += w_image
            image /= tf.math.reduce_max(image)
            attention_maps_batch.write(b, image)
        attention_maps_batch = attention_maps_batch.stack()
        return attention_maps_batch

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
