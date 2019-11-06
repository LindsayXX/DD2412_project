import os
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import tensorflow as tf
import numpy as np
#from PIL import Image
#import skimage
#import tensorflow_datasets as tfds
#import matplotlib.pyplot as plt
from tensorflow.python.keras import backend as K
from tqdm import tqdm
import pathlib

BATCH_SIZE = 32
IMG_HEIGHT = 448
IMG_WIDTH = 448
NUM_CLASSES = 200


class DataSet:

    def __init__(self, path_root):
        self.data_dir = pathlib.Path(path_root + '/CUB_200_2011/CUB_200_2011/images')
        self.image_path = path_root + "/CUB_200_2011/CUB_200_2011/images/"
        self.image_name_path = path_root + "/CUB_200_2011/CUB_200_2011/images.txt"
        self.semantics_path2 = path_root + "/CUB_200_2011/CUB_200_2011/attributes/image_attribute_labels.txt"
        self.semantics_path1 = path_root + "/CUB_200_2011/attributes.txt"
        self.split_path = path_root + "/CUB_200_2011/CUB_200_2011/train_test_split.txt"
        self.class_path = path_root + "/CUB_200_2011/CUB_200_2011/classes.txt"
        self.label_path = path_root + "/CUB_200_2011/CUB_200_2011/image_class_labels.txt"
        self.AUTOTUNE = tf.data.experimental.AUTOTUNE

    def load(self, GPU=True, train=True, batch_size=32):#discard
        index = self.get_split()
        if GPU:
            n = len(index)
        else:
            n = 50
        if train:
            #phi = self.get_phi(index)# Φ, semantic matrix, 28*200
            labels = self.get_label(n, index, set=0)
            images = self.get_image(n, index, set=0)
        else:
            labels = self.get_label(n, index, set=1)
            images = self.get_image(n, index, set=1)
            #phi = self.get_semantic(n, index, set=1) # φ, semantic features 28, n

        ds = tf.data.Dataset.from_tensor_slices((images, np.asarray(labels))).cache().shuffle(50).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

        return ds

    def prepare_for_training(self, ds, batch_size=32, cache=True):
        # This is a small dataset, only load it once, and keep it in memory.
        # use `.cache(filename)` to cache preprocessing work for datasets that don't
        # fit in memory.
        """
        if cache:
            if isinstance(cache, str):
                ds = ds.cache(cache)
            else:
                ds = ds.cache()
        """
        cache_dir = os.path.join(os.getcwd(), 'cache_dir')
        try:
            os.makedirs(cache_dir)
        except OSError:
            print('Cache directory already exists')
        cached = ds.cache(os.path.join(cache_dir, 'cache.temp'))
        ds = ds.shuffle(50).repeat().batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

        return ds

    def get_label(self, n, index, set=0):
        file = open(self.label_path, "r")
        labels = file.readlines()
        label_new = []
        for i in range(n):
            if index[i] == set:
                label_new.append(int(labels[i].split(' ')[1].split('\n')[0]) - 1)# start from 0

        return label_new

    def decode_img(self, img):
        # convert the compressed string to a 3D uint8 tensor
        img = tf.image.decode_jpeg(img, channels=3)
        # Use `convert_image_dtype` to convert to floats in the [0,1] range.
        img = tf.image.convert_image_dtype(img, tf.float32)
        # resize the image to the desired size.
        return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])

    def get_image(self, n, index, set=0):# discard
        images_names = open(self.image_name_path, "r")
        images = images_names.readlines()
        print("loading images...")
        image_new = []
        for i in tqdm(range(n)):
            if index[i] == set:
                im_path = self.image_path + images[i].split(' ')[1].split('\n')[0]
                #img = np.asarray(Image.open(im_path).resize((IMG_WIDTH, IMG_HEIGHT)), dtype=np.float32)
                img = tf.io.read_file(im_path)
                img = self.decode_img(img)
                # image = tf.keras.preprocessing.image.load_img(im_path, target_size=(IMG_WIDTH, IMG_HEIGHT))
                image_new.append(img)
            else:
                pass

        return image_new

    def get_attribute(self):
        file = open(self.semantics_path1, "r")
        lines = file.readlines()
        attributes = {}
        print("loading attributes...")
        for line in lines:
            id = line.split(" ")[0] # No. of attribute, 28 categories, 312 in total
            info = line.split(" ")[1].split("::")
            if info[0] in attributes.keys():
                attributes[info[0]] += [int(id)]
            else:
                attributes[info[0]] = [int(id)]

        return attributes

    def get_semantic(self, n, index, set=0, file_path=None):
        attributes = self.get_attribute()
        n_att = len(attributes.keys())  # 28
        birds_at = {}
        print("loading semantics...")
        file = open(self.semantics_path2, "r")
        lines = file.readlines()
        for line in lines:
            id_bird = line.split(" ")[0]
            if id_bird not in birds_at.keys():
                birds_at[id_bird] = np.zeros(n_att)

            id_att = int(line.split(" ")[1])
            present = int(line.split(" ")[2])
            if present:
                for i, key in enumerate(attributes.keys()):
                    if id_att in attributes[key]:
                        birds_at[id_bird][i] += np.where(np.array(attributes[key]) == id_att)[0][0]

        birds_semantics = []  # 11788*28 list
        for i, key in enumerate(birds_at.keys()):
            if i < n:
                if index[i] == set:
                    birds_semantics.append(birds_at[key])
                else:
                    pass
            else:
                break
        print("Finished!")

        return np.asarray(birds_semantics)

    def get_split(self, index=True):
        file = open(self.split_path, "r")
        ids = file.readlines()
        if index:
            for i in range(len(ids)):#len(set)):
                ids[i] = int(ids[i].split(' ')[1].split('\n')[0])
            return ids
        else:
            images_names = open(self.image_name_path, "r")
            images = images_names.readlines()
            print("splitting...")
            train_list = []
            test_list = []
            for i in range(len(ids)):#len(set)):
                set = int(ids[i].split(' ')[1].split('\n')[0])
                if set == 0:
                    train_list.append(self.image_path + images[i].split(' ')[1].split('\n')[0])
                else:
                    test_list.append(self.image_path + images[i].split(' ')[1].split('\n')[0])

            return tf.data.Dataset.from_tensor_slices(train_list), tf.data.Dataset.from_tensor_slices(test_list)#.cache()

    def get_phi(self):
        index = self.get_split(index=True)
        labels = self.get_label(len(index), index, set=0)
        semantics = self.get_semantic(len(index), index, set=0)
        phi = np.zeros((semantics[0].shape[0], max(labels)+1))
        lcount = {x:labels.count(x) for x in labels}
        for i in range(len(semantics)):
            phi[:, labels[i]] += semantics[i]
        for j in range(phi.shape[0]):
            phi[:, j] = phi[:, j] / lcount[j]

        return tf.convert_to_tensor(phi, dtype=tf.float32)

    def process_path(self, file_path):
        parts = tf.strings.split(file_path, '/')
        # The second to last is the class-directory
        label = int(tf.strings.split(parts[-2], '.')[0])# == self.CLASS_NAMES
        # load the raw data from the file as a string
        img = tf.io.read_file(file_path)
        img = self.decode_img(img)

        return img, label

    def load_gpu(self, batch_size=32):#autotune=4
        # Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
        self.CLASS_NAMES = np.unique(
            np.array([item.name for item in self.data_dir.glob('[!.]*') if item.name != "LICENSE.txt"]))
        train_list_ds, test_list_ds = self.get_split(index=False)
        #dataset = train_list_ds.interleave(tf.data.TFRecordDataset, cycle_length=FLAGS.num_parallel_reads, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        train_ds = train_list_ds.map(self.process_path, num_parallel_calls=self.AUTOTUNE)
        test_ds = test_list_ds.map(self.process_path, num_parallel_calls=self.AUTOTUNE)
        train = self.prepare_for_training(train_ds, batch_size)
        test = self.prepare_for_training(test_ds, batch_size)
        for image, label in train.take(1):
            print("Image shape: ", image.numpy().shape)
            print("Label: ", label.numpy())

        return train, test

    def loadtfds(self, dataset_name, batch_size=32): #not working
        #  Load data from tensorflow_datasets
        raw_train, raw_test = tfds.load(name=dataset_name, split=["train", "test"], batch_size=32)
        train = raw_train.map(lambda x: tf.image.resize(x['image'], (IMG_WIDTH, IMG_HEIGHT)))
        test = raw_test.map(lambda x: tf.image.resize(x['image'], (IMG_WIDTH, IMG_HEIGHT)))
        ds_train = train.shuffle(1000).repeat().batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        ds_test = test.shuffle(1000).repeat().batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        # for batch in ds_train:
        #   ...
        return ds_train, ds_test


if __name__ == '__main__':
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    path_root = os.path.abspath(os.path.dirname(__file__))  # '/content/gdrive/My Drive/data'
    bird_data = DataSet(path_root)
    #train_ds = bird_data.load(GPU=True, train=True, batch_size=32)
    #ds_train, ds_test = bird_data.loadtfds('caltech_birds2011')
    ds_train, ds_test = bird_data.load_gpu(batch_size=4)
    """
    filename1 = 'train_ds.tfrecord'
    writer1 = tf.data.experimental.TFRecordWriter(filename1)
    writer1.write(train_ds)
    #read
    #raw_dataset = tf.data.TFRecordDataset(filenames)
    """
    #image_batch, label_batch = next(iter(ds_train))

