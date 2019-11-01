import IPython.display as display
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import scipy.io as sci
import os
import tensorflow_datasets as tfds
#import tqdm

class DataSet:

    def __init__(self, data_dir, IMG_SIZE=448):
        self.list_ds = tf.data.Dataset.list_files(str(data_dir/'*/*'))
        self.IMG_WIDTH = IMG_SIZE
        self.IMG_HEIGHT = IMG_SIZE

    def get_label(self, file_path):
        # convert the path to a list of path components
        labels = sci.loadmat(file_path)
        parts = tf.strings.split(file_path, '/')
        # The second to last is the class-directory
        #return parts[-2] == CLASS_NAMES

    def decode_img(self,img):
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
        #img = self.decode_img(img)
        # convert the compressed string to a 3D uint8 tensor
        img = tf.image.decode_jpeg(img, channels=3)
        # Use `convert_image_dtype` to convert to floats in the [0,1] range.
        img = tf.image.convert_image_dtype(img, tf.float32)
        # resize the image to the desired size.
        return tf.image.resize(img, [self.IMG_WIDTH, self.IMG_HEIGHT])
        return img, label

    def load(self, autotune=4):
        # Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
        labeled_ds = self.list_ds.map(self.process_path, num_parallel_calls=autotune)
        for image, label in labeled_ds.take(1):
            print("Image shape: ", image.numpy().shape)
            print("Label: ", label.numpy())

def format(image, label):
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1  # normalize?
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    return image, label

def flower(batch_size=32):
    '''
    Load data from tensorflow_datasets
    '''
    raw_train, raw_test = tfds.load(name="oxford_flowers102", split=["train", "test"])
    '''
    for features in raw_train.take(3):
        image, label = features["image"], features["label"]
        plt.imshow(image)
        # plt.imshow(image.numpy()[:, :, 0].astype(np.float32), cmap=plt.get_cmap("gray"))
        print("Label: %d" % label.numpy())
        plt.show()
    '''
    train = raw_train.map(format)
    ds_test = raw_test.map(format)
    ds_train = train.shuffle(1000).batch(batch_size).prefetch(10)  # tf.data.experimental.AUTOTUNE
    # for batch in ds_train:
    #   ...
    return ds_train, ds_test


if __name__=="__main__":
    IMG_SIZE = 448
    this_root = os.path.abspath(os.path.dirname(__file__))
    labels_path = this_root + "/imagelabels.mat"# 102 class, 8189 in total
    setid_path = this_root + "/setid.mat" # setid['trnid']
    image_path = this_root + "/102flowers/jpg"
    flower = DataSet(image_path, IMG_SIZE)
    train_data, test_data = flower.load