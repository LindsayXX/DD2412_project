from tensorflow.keras import layers
import tensorflow as tf
import pathlib
import numpy as np
import matplotlib.pylab as plt

BATCH_SIZE = 32
IMG_SIZE = 448
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
IMG_HEIGHT = 448
IMG_WIDTH = 448

class Kmeans(layers.Layer):

    def __init__(self, clusters_n=2, iterations=10, total_channels=512):
        super(Kmeans, self).__init__()
        self.clusters_n = clusters_n
        self.iterations = iterations
        self.total_channels = total_channels

    def call(self, inputs):
        centroids = tf.Variable(tf.slice(tf.random.shuffle(inputs), [0, 0], [self.clusters_n, -1]))
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
        self.fc1 = tf.keras.layers.Dense(input_shape, activation="relu", kernel_initializer=self.initializer)
        self.fc2 = tf.keras.layers.Dense(input_shape, activation="sigmoid", kernel_initializer=self.initializer)

    def call(self, p_batch):
        out = self.fc1(p_batch)
        out = self.fc2(out)
        a_batch = self.activation(out)
        return a_batch


def get_max_pixels(batch):
    max_batch = []
    for image_batch in batch:
        max_pix = []
        for channel in range(image_batch.shape[2]):
            image = image_batch[:,:,channel]*255
            maximum = tf.math.reduce_max(image)
            x = tf.where(image == maximum)
            if x.shape[0]==1:
                max_pix.append(x)
            else:
                #continue
                choice = tf.dtypes.cast(tf.random.uniform((1,1), 0, x.shape[0]), tf.int32)[0][0]
                max_pix.append(tf.expand_dims(x[choice], 0))
        max_pix = tf.stack(max_pix)
        max_batch.append(max_pix)
    max_batch = tf.stack(max_batch)
    return max_batch

def get_label(file_path):
    # convert the path to a list of path components
    parts = tf.strings.split(file_path, '/')
    # The second to last is the class-directory
    return parts[-2] == CLASS_NAMES

def decode_img(img):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
    # resize the image to the desired size.
    return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])

def process_path(file_path):
    label = get_label(file_path)
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label

def prepare_for_training(ds, cache=True, shuffle_buffer_size=1000):
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
    ds = ds.prefetch(buffer_size=AUTOTUNE)

    return ds



if __name__ == '__main__':
    data_dir = pathlib.Path('/Volumes/Watermelon/CUB_200_2011/CUB_200_2011/images/')
    list_ds = tf.data.Dataset.list_files(str(data_dir / '*/*'))
    CLASS_NAMES = np.unique(np.array([item.name for item in data_dir.glob('[!.]*') if item.name != "LICENSE.txt"]))  # .split('.')[-1]

    # Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    labeled_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)

    train_ds = prepare_for_training(labeled_ds)

    image_batch, label_batch = next(iter(train_ds))

    # first, program the vgg19 pre-trained network
    base_model = tf.keras.applications.VGG19(input_shape=IMG_SHAPE,
                                                   include_top=False,
                                                   weights='imagenet')
    model = tf.keras.Model(inputs=base_model.input, outputs=base_model.get_layer('block3_pool').output)

    feature_batch = base_model(image_batch)

    max_points = get_max_pixels(feature_batch)

    km = Kmeans(clusters_n=2, iterations=10)

    batch_cluster0 = []
    batch_cluster1 = []
    for m in range(max_points.shape[0]):
        cluster0 = []
        cluster1 = []
        assign = km.call(max_points[m, :, 0, :])
        for v in range(assign.shape[0]):
            image = np.array(feature_batch[m, :, :, v])
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

    ap = Average_Pooling()
    p0 = ap.call(batch_cluster0)
    p1 = ap.call(batch_cluster1)

    fc0 = fc(512)
    fc1 = fc(512)
    a0 = fc0.call(p0)
    a1 = fc1.call(p1)

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
