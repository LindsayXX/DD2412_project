import tensorflow as tf
# from tensorflow.python.framework import ops
# from tensorflow.python.framework import dtypes
# from tensorflow.compat.v2.io import decode_image
# from tensorflow.python.ops.gen_io_ops import read_file
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# from tensorflow.keras import optimizers
# from tensorflow.keras.regularizers import l2
# from tensorflow.compat.v2.image import resize
# from tensorflow.compat.v2.image import ResizeMethod
# import matplotlib.pylab as plt
# import numpy as np

class Database():
    def __init__(self):
        self.path = '/Volumes/Watermelon/CUB_200_2011/CUB_200_2011/images/*/*'
        self.IMG_HEIGHT = 448
        self.IMG_WIDTH = 448
        self.IMG_SIZE = 448

    def get_label(self, file_path):
        # convert the path to a list of path components
        parts = tf.strings.split(file_path, '/')
        # The second to last is the class-directory
        return parts[-2]

    def decode_img(self, img):
        # convert the compressed string to a 3D uint8 tensor
        img = tf.image.decode_jpeg(img, channels=3)
        # Use `convert_image_dtype` to convert to floats in the [0,1] range.
        img = tf.image.convert_image_dtype(img, tf.float32)
        # resize the image to the desired size.
        return tf.image.resize(img, [self.IMG_WIDTH, self.IMG_HEIGHT])

    def process_path(self, file_path):
        label = self.get_label(file_path)
        # load the raw data from the file as a string
        img = tf.io.read_file(file_path)
        img = self.decode_img(img)
        return img, label

    def format_example(self, image, label):
        image = tf.cast(image, tf.float32)
        image = (image / 127.5) - 1
        image = tf.image.resize(image, (self.IMG_SIZE, self.IMG_SIZE))
        return image, label

if __name__ == '__main__':
    database = Database()
    list_ds = tf.data.Dataset.list_files(str('/Volumes/Watermelon/CUB_200_2011/CUB_200_2011/images/*/*'))
    labeled_ds = list_ds.map(database.process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    DATASET_SIZE = 37322
    train_size = int(0.7 * DATASET_SIZE)
    val_size = int(0.15 * DATASET_SIZE)
    test_size = int(0.15 * DATASET_SIZE)

    full_dataset = labeled_ds
    raw_train = full_dataset.take(train_size)
    test_dataset = full_dataset.skip(train_size)
    raw_validation = test_dataset.skip(test_size)
    raw_test = test_dataset.take(test_size)

    train = raw_train.map(database.format_example)
    validation = raw_validation.map(database.format_example)
    test = raw_test.map(database.format_example)

    cosa = train.take(1)

    print("Hola")
