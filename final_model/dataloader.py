import os

from tqdm import tqdm
import pathlib
import scipy.io as sio
from sklearn.linear_model import Ridge
import tensorflow as tf
import numpy as np
#import pickle


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
            n = 1000
        if train:
            #phi = self.get_phi(index)# Φ, semantic matrix, 28*200
            labels = self.get_label(n, index, set=0)
            images = self.get_image(n, index, set=0)
        else:
            labels = self.get_label(n, index, set=1)
            images = self.get_image(n, index, set=1)
            #phi = self.get_semantic(n, index, set=1) # φ, semantic features 28, n

        ds = tf.data.Dataset.from_tensor_slices((images, np.asarray(labels))).cache().shuffle(1000).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

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
        ds = ds.shuffle(1000).repeat().batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

        return ds

    def get_label(self, n=11788, index=None, set=0):
        file = open(self.label_path, "r")
        labels = file.readlines()
        if set == 3:
            label_new = np.zeros(n)
        else:
            label_new = []
        for i in range(n):
            if set == 3:
                label_new[i] = int(labels[i].split(' ')[1].split('\n')[0])
            else:
                if index[i] == set:
                    label_new.append(int(labels[i].split(' ')[1].split('\n')[0]))# start from 1

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
        #attributes = self.get_attribute()
        attributes = np.array(range(1, 313))
        n_att = len(attributes)
        #n_att = len(attributes.keys())  # 312
        birds_at = {}
        print("loading semantics...")
        file = open(self.semantics_path2, "r")
        lines = file.readlines()
        for line in lines:
            id_bird = line.split(" ")[0]
            if id_bird not in birds_at.keys():
                birds_at[id_bird] = np.zeros(n_att)

            id_att = int(line.split(" ")[1]) - 1
            birds_at[id_bird][id_att] = int(line.split(" ")[2])
            #present = int(line.split(" ")[2])
            #if present:
            #    for i, key in enumerate(attributes.keys()):
            #        if id_att in attributes[key]:
            #            birds_at[id_bird][i] += np.where(np.array(attributes[key]) == id_att)[0][0]

        birds_semantics = []  # 11788*312 list
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

    def get_split(self, index=True, mode="easy"):
        """
        file = open(self.split_path, "r")
        ids = file.readlines()
        if index:
            for i in range(len(ids)):#len(set)):
                ids[i] = int(ids[i].split(' ')[1].split('\n')[0])
            return ids
        """
        # get the train/test set classes
        mat_fname = 'train_test_split_'
        if mode == "easy":
            inds = sio.loadmat(mat_fname + 'easy')
        else:
            inds = sio.loadmat(mat_fname + 'hard')
        # get the train/test for each id
        index_label = self.get_label(set=3)
        ids = np.zeros(len(index_label))
        classid_train = list(inds['train_cid'][0])
        classid_test = list(inds['test_cid'][0])
        for i in range(len(index_label)):
            if index_label[i] in classid_train:
                ids[i] = 0  # 0 for training
            else:
                ids[i] = 1  # 1 for testing

        if index:
            return ids
        else:
            images_names = open(self.image_name_path, "r")
            images = images_names.readlines()
            print("splitting...")
            train_list = []
            test_list = []
            #for i in range(len(ids)):#len(set)):
            #    set = int(ids[i].split(' ')[1].split('\n')[0])
            for i, set in enumerate(ids):
                if set == 0:
                    train_list.append(self.image_path + images[i].split(' ')[1].split('\n')[0])
                else:
                    test_list.append(self.image_path + images[i].split(' ')[1].split('\n')[0])

            return tf.data.Dataset.from_tensor_slices(train_list), tf.data.Dataset.from_tensor_slices(test_list)#.cache()

    def get_phi(self, set=0):
        # set=0 for training set(seen), set=1 for test set(unseen)
        index = self.get_split(index=True)
        labels = self.get_label(len(index), index, set=set)
        uniq_labels = np.unique(labels)
        sudo_labels = []
        for i in range(len(labels)):
            sudo_labels.append(np.where(labels[i]==uniq_labels)[0][0])

        semantics = self.get_semantic(len(index), index, set=set)
        phi = np.zeros((semantics[0].shape[0], len(uniq_labels)))
        lcount = {x: sudo_labels.count(x) for x in sudo_labels}
        for i in range(len(semantics)):
            phi[:, sudo_labels[i]] += semantics[i]
        for j in range(phi.shape[1]):
            phi[:, j] = phi[:, j] / lcount[j]

        np.save('phi_{}'.format(set), phi)
        return tf.convert_to_tensor(phi, dtype=tf.float32)

    def get_w(self, alpha=1):
        x = self.get_phi(set=0)#phi_seen
        y = self.get_phi(set=1)#phi_unseen
        rr = Ridge(alpha=alpha)
        rr.fit(x, y)
        w = rr.coef_
        np.save("Wseen", w)
        return w

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
        test_ds = test_list_ds.map(self.process_path, num_parallel_calls=self.AUTOTUNE).shuffle(6000)
        valid_ds = test_ds.take(2000).batch(batch_size).repeat().batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        train = self.prepare_for_training(train_ds, batch_size)
        test = self.prepare_for_training(test_ds, batch_size)
        for image, label in train.take(1):
            print("Image shape: ", image.numpy().shape)
            print("Label: ", label.numpy())

        return train, test

    def get_class_split(self, mode="easy"):
        mat_fname = 'train_test_split_'
        if mode == "easy":
            inds = sio.loadmat(mat_fname + 'easy')
        else:
            inds = sio.loadmat(mat_fname + 'hard')
        # get the train/test for each id
        index_label = self.get_label(set=3)
        ids = np.zeros(len(index_label))
        classid_train = list(inds['train_cid'][0])
        classid_test = list(inds['test_cid'][0])
        return classid_train, classid_test


if __name__ == '__main__':
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    path_root = os.path.abspath(os.path.dirname(__file__))  # '/content/gdrive/My Drive/data'
    bird_data = DataSet(path_root)
    # load all imgs
    phi = bird_data.get_phi(set=0)
    w = bird_data.get_w(alpha=0) #(50*150)
    train_class_list, test_class_list = bird_data.get_class_split(mode="easy")
    train_ds, test_ds = bird_data.load_gpu(batch_size=4)
    # only take 1000 images for local test
    # train_ds = bird_data.load(GPU=False, train=True, batch_size=32)
    # test_ds = bird_data.load(GPU=False, train=False, batch_size=32)
    """
    filename1 = 'train_ds.tfrecord'
    writer1 = tf.data.experimental.TFRecordWriter(filename1)
    writer1.write(train_ds)
    #read
    #raw_dataset = tf.data.TFRecordDataset(filenames)
    """
    #image_batch, label_batch = next(iter(ds_train))

