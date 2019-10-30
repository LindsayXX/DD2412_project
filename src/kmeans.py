from tensorflow.keras import layers
import tensorflow as tf
import pathlib
import numpy as np
import matplotlib.pylab as plt
from tensorflow_core.python.keras.models import Sequential

BATCH_SIZE = 32
IMG_SIZE = 448
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
IMG_HEIGHT = 448
IMG_WIDTH = 448


class VGG_feature(layers.Layer):

    def __init__(self):
        super(VGG_feature, self).__init__()

    def call(self):
        base_model = tf.keras.applications.VGG19(input_shape=IMG_SHAPE,
                                                 include_top=False,
                                                 weights='imagenet')
        feature_batch = base_model(image_batch)
        return feature_batch

class Kmeans(layers.Layer):

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
        max_batch = []
        for image_batch in batch:
            max_pix = []
            for channel in range(image_batch.shape[2]):
                image = image_batch[:, :, channel]
                maximum = tf.math.reduce_max(image)
                x = tf.where(image == maximum)
                if x.shape[0] == 1:
                    max_pix.append(x)
                else:
                    # continue
                    choice = tf.dtypes.cast(tf.random.uniform((1, 1), 0, x.shape[0]), tf.int32)[0][0]
                    max_pix.append(tf.expand_dims(x[choice], 0))
            max_pix = tf.stack(max_pix)
            max_batch.append(max_pix)
        max_batch = tf.stack(max_batch)
        return max_batch

    def call(self):
        batch_cluster0 = []
        batch_cluster1 = []
        for m in range(max_points.shape[0]):
            cluster0 = []
            cluster1 = []
            assign = self.cluster(max_points[m, :, 0, :])
            for v in range(assign.shape[0]):
                image = feature_batch[m, :, :, v]
                if assign[v].numpy() == 0:
                    cluster0.append(image)
                    image = tf.zeros((14, 14))
                    cluster1.append(image)
                else:
                    cluster1.append(image)
                    image = tf.zeros((14, 14))
                    cluster0.append(image)

            cluster0 = tf.stack(cluster0)
            cluster1 = tf.stack(cluster1)
            batch_cluster0.append(cluster0)
            batch_cluster1.append(cluster1)
        return batch_cluster0, batch_cluster1

class Average_Pooling(layers.Layer):

    def __init__(self):
        super( Average_Pooling, self).__init__()

    def call(self, cluster):
        p_batch = []
        for b in cluster:
            H, W = b.shape[1], b.shape[2]
            p = tf.math.reduce_sum(b, axis=(1, 2))/(H*W)
            p_batch.append(p)
        return p_batch


class fc(layers.Layer):

    def __init__(self, input_shape):
        super(fc, self).__init__()
        self.initializer = tf.keras.initializers.glorot_normal()
        self.fc1 = tf.keras.layers.Dense(input_shape, input_shape=(input_shape,), activation="relu", kernel_initializer=self.initializer)
        self.fc2 = tf.keras.layers.Dense(input_shape, activation="sigmoid", kernel_initializer=self.initializer)
        self.bn = tf.keras.layers.BatchNormalization()

    def call(self, p_batch):
        a_batch = []
        for p in p_batch:
            p = tf.expand_dims(p, 0)
            out = self.fc1(p)
            out = self.fc2(out)
            a = self.bn(out)
            a_batch.append(a)
        a_batch = tf.squeeze(tf.stack(a_batch))
        return a_batch


class weighted_sum(layers.Layer):

    def __init__(self):
        super(weighted_sum, self).__init__()

    def call(self, batch, a_batch):
        attention_maps_batch = []
        for b in range(batch.shape[0]):
            image = tf.zeros((batch.shape[1], batch.shape[2]))
            for c in range(batch.shape[-1]):
                w_image = tf.multiply(batch[b, :, :, c], a_batch[b, c])
                image += w_image
            image /= tf.math.reduce_max(image)
            attention_maps_batch.append(image)
        attention_maps_batch = tf.stack(attention_maps_batch)
        return attention_maps_batch


class Database():

    def __init__(self):
        self.data_dir = pathlib.Path('/Volumes/Watermelon/CUB_200_2011/CUB_200_2011/images/')
        self.AUTOTUNE = tf.data.experimental.AUTOTUNE

    def call(self):
        list_ds = tf.data.Dataset.list_files(str(self.data_dir / '*/*'))
        self.CLASS_NAMES = np.unique(
            np.array([item.name for item in self.data_dir.glob('[!.]*') if item.name != "LICENSE.txt"]))

        labeled_ds = list_ds.map(self.process_path, num_parallel_calls=self.AUTOTUNE)

        train_ds = self.prepare_for_training(labeled_ds)

        image_batch, label_batch = next(iter(train_ds))

        return image_batch, label_batch

    def get_label(self, file_path):
        # convert the path to a list of path components
        parts = tf.strings.split(file_path, '/')
        # The second to last is the class-directory
        return parts[-2] == self.CLASS_NAMES

    def decode_img(self, img):
        # convert the compressed string to a 3D uint8 tensor
        img = tf.image.decode_jpeg(img, channels=3)
        # Use `convert_image_dtype` to convert to floats in the [0,1] range.
        img = tf.image.convert_image_dtype(img, tf.float32)
        # resize the image to the desired size.
        return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])

    def process_path(self, file_path):
        label = self.get_label(file_path)
        # load the raw data from the file as a string
        img = tf.io.read_file(file_path)
        img = self.decode_img(img)
        return img, label

    def prepare_for_training(self, ds, cache=True, shuffle_buffer_size=1000):
        # This is a small dataset, only load it once, and keep it in memory.
        # use `.cache(filename)` to cache preprocessing work for datasets that don't
        # fit in memory.
        if cache:
            if isinstance(cache, str):
                ds = ds.cache(cache)
            else:
                ds = ds.cache()

        ds = ds.shuffle(buffer_size=shuffle_buffer_size)

        # Repeat forever
        ds = ds.repeat()

        ds = ds.batch(BATCH_SIZE)

        # `prefetch` lets the dataset fetch batches in the background while the model
        # is training.
        ds = ds.prefetch(buffer_size=self.AUTOTUNE)

        return ds

if __name__ == '__main__':
    data = Database()
    image_batch, label_batch = data.call()

    # first, program the vgg19 pre-trained network

    vgg_features = VGG_feature()
    feature_batch = vgg_features.call()

    km = Kmeans(clusters_n=2, iterations=10)
    max_points = km.get_max_pixels(feature_batch)
    batch_cluster0, batch_cluster1 = km.call()

    ap = Average_Pooling()
    p0 = ap.call(batch_cluster0)
    p1 = ap.call(batch_cluster1)

    fc0 = fc(512)
    fc1 = fc(512)
    a0 = fc0.call(p0)
    a1 = fc1.call(p1)

    ws = weighted_sum()
    M0 = ws.call(feature_batch, a0)
    M1 = ws.call(feature_batch, a1)

    print("hola")
    # for m in range(max_points.shape[0]):
    #     assign = km.call(max_points[m, :, 0, :])
    #     image0 = np.zeros((14, 14))
    #     image1 = np.zeros((14, 14))
    #     for v in range(assign.shape[0]):
    #
    #         image = np.array(feature_batch[m, :, :, v])
    #         if assign[v].numpy() == 1:
    #             image1 += image
    #         elif assign[v].numpy() == 0:
    #             image0 += image
    #     image1 /= np.max(image1)
    #     image0 /= np.max(image0)
    #     fig, ax = plt.subplots(1, 2)
    #     ax[0].imshow(image1)
    #     ax[1].imshow(image0)
    #     plt.show()
    # with tf.GradientTape() as tape:
