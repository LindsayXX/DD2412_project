from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay
from Classification import Classifier, Classifier_Unseen
from Multi_attention_subnet import Kmeans, WeightedSum
from Cropping_subnet import RCN, Crop
import os
from dataloader import DataSet

CHANNELS = 512
BATCH_SIZE = 32
N_CLASSES = 150
SEMANTIC_SIZE = 312
IMG_SIZE = 448
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)


class FinalModel(tf.keras.Model):

    def __init__(self):
        super(FinalModel, self).__init__()

        # MULTI ATTENTION SUBNET
        self.vgg_features_initial = tf.keras.applications.VGG19(input_shape=IMG_SHAPE,
                                                                include_top=False,
                                                                weights='imagenet')  # VGG_feature()
        self.vgg_features_initial.trainable = False
        self.kmeans = Kmeans(clusters_n=2, iterations=10, batch_size=BATCH_SIZE)
        self.average_pooling_0 = tf.keras.layers.GlobalAveragePooling2D()
        self.average_pooling_1 = tf.keras.layers.GlobalAveragePooling2D()
        self.fc1_1 = tf.keras.layers.Dense(CHANNELS, input_shape=(512,), activation="relu")
        self.fc1_2 = tf.keras.layers.Dense(CHANNELS, input_shape=(512,), activation="relu")
        self.fc2_1 = tf.keras.layers.Dense(CHANNELS, activation="sigmoid")
        self.fc2_2 = tf.keras.layers.Dense(CHANNELS, activation="sigmoid")

        self.bn0 = tf.keras.layers.BatchNormalization()
        self.bn1 = tf.keras.layers.BatchNormalization()

        self.weighted_sum0 = WeightedSum()
        self.weighted_sum1 = WeightedSum()
        self.c_init = tf.random_normal_initializer(0, 0.01)
        self.C = tf.Variable(initial_value=self.c_init(shape=(N_CLASSES, SEMANTIC_SIZE),
                                                       dtype=tf.float32), trainable=True)

        # cropping net
        self.crop_net0 = RCN(hidden_unit=14, map_size=14, image_size=448)
        self.crop_net1 = RCN(hidden_unit=14, map_size=14, image_size=448)
        self.crop0 = Crop()
        self.crop1 = Crop()

        # joint feature learning subnet
        '''shit'''
        '''
        self.reshape_global = ReShape224()
        self.reshape_local0 = ReShape224()
        self.reshape_local1 = ReShape224()
        self.vgg_features_global = tf.keras.applications.VGG19(input_shape=(224, 224, 3),
                                                               include_top=False,
                                                               weights='imagenet')  # VGG_feature()
        self.vgg_features_global.trainable = False
        self.vgg_features_local0 = tf.keras.applications.VGG19(input_shape=(224, 224, 3),
                                                               include_top=False,
                                                               weights='imagenet')  # VGG_feature()
        self.vgg_features_local0.trainable = False
        self.vgg_features_local1 = tf.keras.applications.VGG19(input_shape=(224, 224, 3),
                                                               include_top=False,
                                                               weights='imagenet')  # VGG_feature()
        self.vgg_features_local1.trainable = False
        self.global_score = Scores()
        self.score0 = Scores()
        self.score1 = Scores()
        '''
        self.vgg_features_global = tf.keras.applications.VGG19(input_shape=(224, 224, 3), include_top=False,
                                                               weights='imagenet', pooling="avg")
        self.W = tf.keras.layers.Dense(SEMANTIC_SIZE, activation="relu")
        self.l2loss = tf.keras.layers.Lambda(
            lambda x: tf.keras.backend.sum(tf.keras.backend.square(x[0] - x[1][:, 0]), 1, keepdims=True))
        #self.fc = tf.keras.layers.Dense(N_CLASSES, activation="softmax")
        # trainable class centers, C={c1, c2, ..., c_{n_classes}}
        c_init = tf.random_normal_initializer(0, 0.01)
        self.C = tf.Variable(initial_value=c_init(shape=(SEMANTIC_SIZE, N_CLASSES), dtype='float32'), trainable=True,
                             name="C")
        self.classifier = Classifier(N_CLASSES)

    def call(self, x, Phi):
        # MULTI ATTENTION SUBNET
        # x will be image_batch of shape (BATCH,448,448,3)
        print("VGG")
        feature_map = self.vgg_features_initial(x)  # gives an output of shape (BATCH,14,14,512)
        print("KMEANS")
        batch_cluster0, batch_cluster1 = self.kmeans(feature_map)  # gives two lists containing tensors of shape (512,14,14)
        print("AVR POOL")
        p1 = self.average_pooling_0(batch_cluster0)  # gives a list of length=batch_size containing tensors of shape (512,)
        p2 = self.average_pooling_1(batch_cluster1)  # gives a list of length=batch_size containing tensors of shape (512,)
        del batch_cluster0
        del batch_cluster1

        print("FC")
        out0 = self.fc1_1(p1)
        out1 = self.fc1_2(p2)

        out0 = self.fc2_1(out0)
        out1 = self.fc2_2(out1)

        a0 = self.bn0(out0)
        a1 = self.bn1(out1)
        del out0
        del out1

        print("WS")
        m0 = self.weighted_sum0(feature_map, a0)  # gives tensor of shape (BATCH,14,14)
        m1 = self.weighted_sum1(feature_map, a1)  # gives tensor of shape (BATCH,14,14)

        print("CROP")
        # CROPPING SUBNET
        mask0 = self.crop_net0(m0)  # shape(BATCH,14,14)
        mask1 = self.crop_net1(m1)  # shape(BATCH,14,14)
        croped0, newmask1 = self.crop0(x, mask0)  # of shape (BATCH,448,448,3)
        croped1, newmask2 = self.crop1(x, mask1)  # of shape (BATCH,448,448,3)
        del newmask1
        del newmask2

        print("RESIZE")
        # resizing the outputs of cropping network
        full_image = tf.image.resize(x, (224, 224)) #self.reshape_global(x)
        attended_part0 = tf.image.resize(croped0, (224, 224)) #self.reshape_local0(croped0)
        attended_part1 = tf.image.resize(croped1, (224, 224)) #self.reshape_local1(croped1)
        del croped0
        del croped1

        print("JFL")
        # feeding the 3 images into VGG nets - get the visual feature theta
        global_theta = self.vgg_features(full_image)
        local_theta0 = self.vgg_features(attended_part0)
        local_theta1 = self.vgg_features(attended_part1)

        # computing the mapped features(phi), should have size 1 * n_semantic
        global_phi = self.W(global_theta)
        local0_phi = self.W(local_theta0)
        local1_phi = self.W(local_theta1)

        # compatibility score: s_j^i = theta_i(x)^T W_i phi(y_i), should have size 1 * n_classes
        global_scores = tf.linalg.matmul(global_phi, Phi)
        local_scores0 = tf.linalg.matmul(global_phi, Phi)
        local_scores1 = tf.linalg.matmul(global_phi, Phi)

        # sum (and normalize?) the scores and mapped features
        score = tf.add(tf.add(global_theta, local_scores0), local_scores1)#tf.math.reduce_sum(, axis=1, keepdims=True)
        # avg_score = tf.multiply(sum_gll, 1.0 / 3.0)
        # compute ||~phi_i - ~Ci|| and ||~phi_i - ~Cj||, '~' is normalization
        # l2loss = self.l2loss(
        #    [tf.math.l2_normalize(tf.squeeze(out)), tf.math.l2_normalize(tf.transpose(center, perm=[0, 2, 1]))])
        phi_mapped = tf.add(tf.add(global_phi, local0_phi), local1_phi)
        #avg_phi = tf.multiply(sum_gll, 1.0 / 3.0)

        y_pred = self.classifier(score)

        return m0, m1, mask0, mask1, score, phi_mapped, y_pred, self.C

if __name__ == '__main__':
    # just for testing
    path_root = os.path.abspath(os.path.dirname(__file__))  # '/content/gdrive/My Drive/data'
    bird_data = DataSet("D:/MY2/ADDL/DD2412_project/basemodel")
    PHI = bird_data.get_phi(set=0)
    #w = bird_data.get_w(alpha=1)  # (50*150)
    #train_class_list, test_class_list = bird_data.get_class_split(mode="easy")
    # only take 1000 images for local test
    train_ds = bird_data.load(GPU=False, train=True, batch_size=4)
    # test_ds = bird_data.load(GPU=False, train=False, batch_size=32)
    image_batch, label_batch = next(iter(train_ds))
    test_model = FinalModel()
    m0, m1, mask0, mask1, scores, phi, y_pred, C = test_model(image_batch, PHI)
