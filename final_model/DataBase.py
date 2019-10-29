import tensorflow as tf
import pathlib
import numpy as np

BATCH_SIZE = 32
IMG_HEIGHT = 448
IMG_WIDTH = 448


class Database():

    def __init__(self):
        self.data_dir = pathlib.Path('/Users/stella/Downloads/CUB_200_2011/CUB_200_2011/images')
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


