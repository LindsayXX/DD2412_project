from __future__ import absolute_import, division, print_function, unicode_literals

import os

import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Activation, GlobalAveragePooling2D
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
import matplotlib.pyplot as plt
import numpy as np

from Classification import Classifier
from Multi_attention_subnet import VGG_feature, Kmeans, Average_Pooling, Fc, WeightedSum
from Cropping_subnet import ReShape224, RCN, Crop
from Joint_feature_learning_subnet import Scores, JFL
from dataloader import DataSet
import tensorflow.keras.optimizers as opt

from Losses import Loss

# IMG_SIZE = 448
CHANNELS = 512
BATCH_SIZE = 5
N_CLASSES = 200
SEMANTIC_SIZE = 28
IMG_SIZE = 448
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)


class FinalModel(Model):

    def __init__(self):
        super(FinalModel, self).__init__()

        # MULTI ATTENTION SUBNET
        self.vgg_features_initial = tf.keras.applications.VGG19(input_shape=IMG_SHAPE,
                                                                include_top=False,
                                                                weights='imagenet')  # VGG_feature()
        self.vgg_features_initial.trainable = False
        self.kmeans = Kmeans(clusters_n=2, iterations=10)
        self.average_pooling_0 = Average_Pooling()
        self.average_pooling_1 = Average_Pooling()
        """
        self.fc_0 = Fc(CHANNELS)
        self.fc_1 = Fc(CHANNELS)
        """
        self.initializer = tf.keras.initializers.glorot_normal()
        self.fc1_1 = tf.keras.layers.Dense(512, input_shape=(512,), activation="relu",
                                           kernel_initializer=self.initializer)
        self.fc1_2 = tf.keras.layers.Dense(512, input_shape=(512,), activation="relu",
                                           kernel_initializer=self.initializer)

        self.fc2_1 = tf.keras.layers.Dense(512, activation="sigmoid", kernel_initializer=self.initializer)
        self.fc2_2 = tf.keras.layers.Dense(512, activation="sigmoid", kernel_initializer=self.initializer)

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

        self.average_pooling_global = tf.keras.layers.GlobalAveragePooling2D()  # Average_Pooling()
        self.average_pooling_local0 = tf.keras.layers.GlobalAveragePooling2D()  # Average_Pooling()
        self.average_pooling_local1 = tf.keras.layers.GlobalAveragePooling2D()  # Average_Pooling()

        self.joint_net_global = JFL(N_CLASSES, SEMANTIC_SIZE, CHANNELS)
        self.joint_net_local0 = JFL(N_CLASSES, SEMANTIC_SIZE, CHANNELS)
        self.joint_net_local1 = JFL(N_CLASSES, SEMANTIC_SIZE, CHANNELS)

        self.global_score = Scores()
        self.score0 = Scores()
        self.score1 = Scores()

        self.classifier = Classifier(N_CLASSES)

    def call(self, x, phi):
        # MULTI ATTENTION SUBNET
        # x will be image_batch of shape (BATCH,448,448,3)
        print("VGG")
        feature_map = self.vgg_features_initial(x)  # gives an output of shape (BATCH,14,14,512)
        print(feature_map.shape)
        print("KMEANS")
        batch_cluster0, batch_cluster1 = self.kmeans(feature_map)  # gives two lists containing tensors of shape (512,14,14)
        print("AVR POOL")
        print(batch_cluster0.shape, batch_cluster1.shape)
        p1 = self.average_pooling_0(batch_cluster0)  # gives a list of length=batch_size containing tensors of shape (512,)
        p2 = self.average_pooling_1(batch_cluster1)  # gives a list of length=batch_size containing tensors of shape (512,)
        print("FC")

        """
        a0 = self.fc_0(p1)  # gives tensor of shape (BATCH,512)
        a1 = self.fc_1(p2)  # gives tensor of shape (BATCH,512)
        """
        out0 = self.fc1_1(p1)
        out1 = self.fc1_2(p2)

        out0 = self.fc2_1(out0)
        out1 = self.fc2_2(out1)

        a0 = self.bn0(out0)
        a1 = self.bn1(out1)

        print("WS")
        m0 = self.weighted_sum0(feature_map, a0)  # gives tensor of shape (BATCH,14,14)
        m1 = self.weighted_sum1(feature_map, a1)  # gives tensor of shape (BATCH,14,14)

        # m0 = tf.convert_to_tensor(np.load("im0.npy"))
        # m1 = tf.convert_to_tensor(np.load("im1.npy"))
        attmap_out = tf.TensorArray(tf.float32, 2)
        attmap_out.write(0, m0)
        attmap_out.write(1, m1)
        attmap_out = attmap_out.stack()

        print("CROP")
        # CROPPING SUBNET
        mask0 = self.crop_net0(m0)  # shape(BATCH,14,14)
        mask1 = self.crop_net1(m1)  # shape(BATCH,14,14)
        croped0, newmask1 = self.crop0(x, mask0)  # of shape (BATCH,448,448,3)
        croped1, newmask2 = self.crop1(x, mask1)  # of shape (BATCH,448,448,3)
        # attention_crop_out = tf.TensorArray(tf.float32, size=2)
        # attention_crop_out.write(0, mask0)
        # attention_crop_out.write(1, mask1)
        # attention_crop_out = attention_crop_out.stack()
        # gives 3 tensors of size (batch,448,448,3)

        print("RESHAPE")
        # JOINT FEATURE LEARNING SUBNE
        # resizing the outputs of cropping network
        full_image = self.reshape_global(x)
        attended_part0 = self.reshape_local0(croped0)
        attended_part1 = self.reshape_local1(croped1)

        # print("VGG")
        # feeding the 3 images into VGG nets
        full_image = self.vgg_features_global(full_image)
        attended_part0 = self.vgg_features_local0(attended_part0)
        attended_part1 = self.vgg_features_local1(attended_part1)
        # print(full_image.shape, attended_part0.shape, attended_part1.shape)

        # print("THETAS")
        # creating thetas
        global_theta = self.average_pooling_global(full_image)
        local_theta0 = self.average_pooling_local0(attended_part0)
        local_theta1 = self.average_pooling_local1(attended_part1)

        # np.save("theta_global", np.array(global_theta))
        # np.save("theta_local0", np.array(local_theta0))
        # np.save("theta_local1", np.array(local_theta1))
        # np.save("phi", np.array(phi))

        # computing the scores
        # print("Computing Scores...")
        global_scores, global_phi = self.joint_net_global.call(global_theta,
                                                               phi)  # self.global_score(global_theta, phi)
        local_scores0, local0_phi = self.joint_net_local0.call(local_theta0, phi)  # self.score0(local_theta0, phi)
        local_scores1, local1_phi = self.joint_net_local1.call(local_theta1, phi)  # self.score1(local_theta1, phi)
        # scores_out = tf.TensorArray(tf.float32, 3)
        # scores_out.write(0, global_scores)
        # scores_out.write(1, local_scores0)
        # scores_out.write(2, local_scores1)
        # scores_out = scores_out.stack()
        #phis_out = tf.TensorArray(tf.float32, 3)
        #phis_out.write(0, global_phi)
        #phis_out.write(1, local0_phi)
        #phis_out.write(2, local1_phi)
        #phis_out = phis_out.stack()

        # print(scores_out.shape)
        y_pred = self.classifier(global_scores, local_scores0, local_scores1)

        return m0, m1, mask0, mask1, global_scores, local_scores0, local_scores1, \
               global_phi, local0_phi, local1_phi, y_pred, self.C


@tf.function
def train_step(model, image_batch, y_true, PHI, loss_fun, opt_fun):
    with tf.GradientTape() as tape:
        m0, m1, mask0, mask1, global_scores, local_scores0, local_scores1, global_phi, local0_phi, local1_phi, y_pred, C = model(image_batch, PHI)
        loss = loss_fun(m0, m1, mask0, mask1, global_scores, local_scores0, local_scores1, global_phi, local0_phi, local1_phi, y_true, y_pred,
                        N_CLASSES, image_batch.shape[0], C)
    gradients = tape.gradient(loss, model.trainable_variables)
    opt_fun.apply_gradients(zip(gradients, model.trainable_variables))
    print("Current TrainLoss: {}".format(loss))
    train_loss(loss)
    print(labels.shape, y_pred.shape)
    train_accuracy(tf.expand_dims(labels, -1), tf.expand_dims(y_pred, -1))

# test the model
# @tf.function
def test_step(model, images, loss_fun):
    m_i, m_k, map_att, gtmap, score, y_true, y_pred, n_classes, batch_size = model(images)
    loss = loss_fun(m_i, m_k, map_att, gtmap, score, y_true, y_pred, n_classes, batch_size)
    print("Current TestLoss: {}".format(loss))


# testing by running

if __name__ == '__main__':

    tf.compat.v1.enable_eager_execution()
    gpu = tf.config.experimental.list_physical_devices('GPU')
    print("Num GPUs Available: ", len(gpu))
    if gpu:
        tf.config.experimental.set_memory_growth(gpu, True)
    path_root = os.path.abspath(os.path.dirname(__file__))
    database = DataSet("/Volumes/Watermelon")#path_root)
    PHI = database.get_phi()
    DS, DS_test = database.load_gpu(batch_size=32)  # image_batch, label_batch
    modelaki = FinalModel()

    loss_fun = Loss().final_loss
    opt_fun = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9)

    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=opt_fun, net=modelaki)
    manager = tf.train.CheckpointManager(ckpt, path_root + '/tf_ckpts',
                                         max_to_keep=3)  # keep only the three most recent checkpoints
    ckpt.restore(manager.latest_checkpoint)  # pickup training from where you left off

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.Accuracy(name='train_accuracy')
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.Accuracy(name='test_accuracy')

    # m0, m1 = modelaki.call(image_batch)

    EPOCHS = 50
    CHECKEPOCHS = 1

    # train_loss = tf.keras.metrics.Mean(name='train_loss')
    # image_batch, label_batch = next(iter(DS))
    for epoch in range(EPOCHS):
        train_loss_results = []
        train_accuracy_results = []
        for images, labels in DS:
            if images.shape[0] == BATCH_SIZE:
                train_step(modelaki, images, labels, PHI, loss_fun, opt_fun)
                tf.print('Epoch {}, train_Loss: {}, train_Accuracy: {}\n'.format(epoch + 1, train_loss.result(),
                                                                                 train_accuracy.result()))
                train_loss_results.append(train_loss.result())
                train_accuracy_results.append(train_accuracy.result())
        ckpt.step.assign_add(1)
        if int(ckpt.step) % CHECKEPOCHS == 0:
            save_path = manager.save()
            with open(path_root + '/log.txt', 'a') as temp:
                temp.write('Epoch {}, train_Loss: {}, train_Accuracy: {}\n'.format(
                    epoch + 1, sum(train_loss_results) / len(train_accuracy_results),
                    sum(train_accuracy_results) / len(train_accuracy_results)))
                # , test_loss.result(), test_accuracy.result()))

        template = 'Epoch {}, Loss: {}'
        print(template.format(epoch + 1, train_loss))

# TODO: pre-train vgg only on birds
# TODO: global variables of this module may differ from the subnetworks module.
