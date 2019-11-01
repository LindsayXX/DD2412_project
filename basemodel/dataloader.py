import tensorflow as tf
import pathlib
import numpy as np
import os
from PIL import Image
import skimage
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from tensorflow.python.keras import backend as K
from tqdm import tqdm

BATCH_SIZE = 32
IMG_HEIGHT = 448
IMG_WIDTH = 448
NUM_CLASSES = 200


class DataSet:

    def __init__(self, path_root):
        self.image_path = path_root + "/CUB_200_2011/CUB_200_2011/images/"
        self.image_name_path = path_root + "/CUB_200_2011/CUB_200_2011/images.txt"
        self.semantics_path2 = path_root + f"/CUB_200_2011/CUB_200_2011/attributes/image_attribute_labels.txt"
        self.semantics_path1 = path_root + "/CUB_200_2011/attributes.txt"
        self.split_path = path_root + f"/CUB_200_2011/CUB_200_2011/train_test_split.txt"
        self.class_path = path_root + "/CUB_200_2011/CUB_200_2011/classes.txt"
        self.label_path = path_root + "/CUB_200_2011/CUB_200_2011/image_class_labels.txt"
        self.AUTOTUNE = tf.data.experimental.AUTOTUNE

    def load(self, GPU=True, train=True, batch_size=32):#discard
        index = self.get_split()
        if GPU:
            n = len(index)
        else:
            n = 1000
        if train:
            phi = self.get_phi(index)# Φ, semantic matrix, 28*200
            labels = self.get_label(n, index, set=0)
            images = self.get_image(n, index, set=0)
        else:
            labels = self.get_label(n, index, set=1)
            images = self.get_image(n, index, set=1)
            phi = self.get_semantic(n, index, set=1) # φ, semantic features 28, n

        ds = tf.data.Dataset.from_tensor_slices((images, np.asarray(labels))).shuffle(1000).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

        return ds, tf.convert_to_tensor(phi, dtype=tf.float32)

    def prepare_for_training(self, ds, cache=True, shuffle_buffer_size=1000):
        # This is a small dataset, only load it once, and keep it in memory.
        # use `.cache(filename)` to cache preprocessing work for datasets that don't
        # fit in memory.
        if cache:
            if isinstance(cache, str):
                ds = ds.cache(cache)
            else:
                ds = ds.cache()

    def get_label(self, n, index, set=0):
        file = open(self.label_path, "r")
        labels = file.readlines()
        label_new = []
        for i in range(n):
            if index[i] == set:
                label_new.append(int(labels[i].split(' ')[1].split('\n')[0]) - 1)# start from 0

        return label_new

    def decode_img(self,img):
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

    def get_split(self):
        file = open(self.split_path, "r")
        set = file.readlines()
        for i in range(len(set)):#len(set)):
            set[i] = int(set[i].split(' ')[1].split('\n')[0])

        return set

    def get_phi(self, index):
        labels = self.get_label(len(index), index, set=0)
        semantics = self.get_semantic(len(index), index, set=0)
        phi = np.zeros((semantics[0].shape[0], max(labels)+1))
        lcount = {x:labels.count(x) for x in labels}
        for i in range(len(semantics)):
            phi[:, labels[i]] += semantics[i]
        for j in range(phi.shape[0]):
            phi[:, j] = phi[:, j] / lcount[j]

        return phi

    '''
    def loadtfds(self, datas{x:labels.count(x) for x in labels}et_name, batch_size=32):
        #Load data from tensorflow_datasets
        raw_train, raw_test = tfds.load(name=dataset_name, split=["train", "test"])
        image_train = image_train.map(tf.image.resize(image, (IMG_WIDTH, IMG_HEIGHT)) for image in image_train)
        image_test = image_test.map(tf.image.resize(image, (IMG_WIDTH, IMG_HEIGHT)) for image in image_test)
        index = self.get_split()
        semantic_train = self.get_semantic(index, set=0)
        semantic_test = self.get_split(index, set=1)
        train = tf.data.Dataset.zip((image_train, tf.data.Dataset.from_tensor_slices(semantic_train)))
        ds_train = train.shuffle(1000).batch(batch_size).prefetch(10)  # tf.data.experimental.AUTOTUNE
        test = tf.data.Dataset.zip((image_test, tf.data.Dataset.from_tensor_slices(semantic_test)))
        ds_test = test.shuffle(1000).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        # for batch in ds_train:
        #   ...
        return ds_train, ds_test
    '''
