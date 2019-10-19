# import IPython.display as display
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os

# from tensorflow_core.python.framework import ops
# from tensorflow_core.python.framework import dtypes

# from tensorflow_core.python.ops.gen_image_ops import decode_png
# from tensorflow_core.python.ops.gen_io_ops import read_file


class Bird:

    def __init__(self):
        label = None
        image = None
        parts = None
        id = None

class Database:

    def __init__(self, parent_path):
        self.parent_path = parent_path
        self.birds = {}
        self.triplets = {}

    def create_triplets(self, label_file=None, images_file=None):
        """
        Function in charge of creating the triplets (id, label, image_path) from the dataset
        :param label_file: txt file which contains the labels of each image
        :param images_file: txt file which contains the path of each image
        :return:
        """
        id = []
        labels = []
        im_paths = []

        with open(self.parent_path+label_file, 'r') as File:
            infoFile = File.readlines()
            for line in infoFile:
                words = line.split()
                id.append(int(words[0]))
                im_paths.append(self.parent_path + "images/"+ words[1])
        with open(self.parent_path+images_file, 'r') as File:
            infoFile = File.readlines()
            for line in infoFile:
                words = line.split()
                labels.append(int(words[1]))

        NumFiles = len(labels)
        tim_paths = tf.convert_to_tensor(im_paths)
        tlabels = tf.convert_to_tensor(labels)
        tid = tf.convert_to_tensor(id)

        for i in range(NumFiles):
            self.triplets[i] = (tid[i], tlabels[i], tim_paths[i])

    def read_batch(self, init, end):
        """
        Function in charge of creating a batch of images from information
        in triplets
        :param init: index to the first value to extract from triplets
        :param end: index to the last value to extract from triplets
        :return:
        """
        batch = {}
        ind_batch = 0
        for i in np.arange(init, end):
            b = Bird()
            im_path = self.triplets[i][2]
            rawIm = tf.io.read_file(im_path)
            b.image = tf.io.decode_png(rawIm)
            b.label = self.triplets[i][1]
            b.id = self.triplets[i][0]
            batch[ind_batch] = b
            ind_batch += 1
        return batch

if __name__ == '__main__':
    parent_path = "/Volumes/Watermelon/CUB_200_2011/CUB_200_2011/"
    db = Database(parent_path)
    db.create_triplets(label_file="images.txt", images_file="image_class_labels.txt")
    batch = db.read_batch(10, 30)
    print("hola pianola")
