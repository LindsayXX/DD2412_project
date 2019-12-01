from __future__ import absolute_import, division, print_function, unicode_literals

import datetime
import os

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import Model
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay
from Classification import Classifier, Classifier_Unseen
from Multi_attention_subnet import VGG_feature, Kmeans, Average_Pooling, Fc, WeightedSum
from Cropping_subnet import ReShape224, RCN, Crop
from Joint_feature_learning_subnet import Scores
from dataloader import DataSet

from Losses import Loss

# IMG_SIZE = 448
import sys
sys.path.append("../src")
from jointmodel import JFL

CHANNELS = 512
BATCH_SIZE = 32
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

        print("RESHAPE")
        # JOINT FEATURE LEARNING SUBNE
        # resizing the outputs of cropping network
        full_image = self.reshape_global(x)
        attended_part0 = self.reshape_local0(croped0)
        attended_part1 = self.reshape_local1(croped1)
        del croped0
        del croped1

        # print("VGG")
        # feeding the 3 images into VGG nets
        full_image = self.vgg_features_global(full_image)
        attended_part0 = self.vgg_features_local0(attended_part0)
        attended_part1 = self.vgg_features_local1(attended_part1)

        # print("THETAS")
        # creating thetas
        global_theta = self.average_pooling_global(full_image)
        local_theta0 = self.average_pooling_local0(attended_part0)
        local_theta1 = self.average_pooling_local1(attended_part1)

        # computing the scores
        # print("Computing Scores...")
        global_scores, global_phi = self.joint_net_global.call(global_theta, phi)  # self.global_score(global_theta, phi)
        local_scores0, local0_phi = self.joint_net_local0.call(local_theta0, phi)  # self.score0(local_theta0, phi)
        local_scores1, local1_phi = self.joint_net_local1.call(local_theta1, phi)  # self.score1(local_theta1, phi)

        # average scores
        sum_gl = tf.add(global_scores, local_scores0)
        sum_gll = tf.add(sum_gl, local_scores1)
        #avg_score = tf.multiply(sum_gll, 1.0 / 3.0)

        # average phi
        sum_gl = tf.add(global_phi, local0_phi)
        sum_gll = tf.add(sum_gl, local1_phi)
        #avg_phi = tf.multiply(sum_gll, 1.0 / 3.0)

        # average
        y_pred = self.classifier(sum_gll)

        return m0, m1, mask0, mask1, sum_gll, sum_gll, y_pred, self.C


# @tf.function
# def train_step(model, image_batch, y_true, PHI, loss_fun, opt_fun, epoch):
#     with tf.GradientTape() as tape:
#         m0, m1, mask0, mask1, global_scores, local_scores0, local_scores1, global_phi, local0_phi, local1_phi, y_pred, C = model(image_batch, PHI)
#         loss = loss_fun(m0, m1, mask0, mask1, global_scores, local_scores0, local_scores1, global_phi, local0_phi, local1_phi, y_true, y_pred,
#                         N_CLASSES, image_batch.shape[0], C)
#     gradients = tape.gradient(loss, model.trainable_variables)
#     opt_fun.apply_gradients(zip(gradients, model.trainable_variables))
#
#     # summary for tensorboard
#     with train_summary_writer.as_default():
#         tf.summary.scalar('loss', loss.result(), step=epoch)
#         tf.summary.scalar('accuracy', train_accuracy(tf.expand_dims(y_true, -1),
#                                                      tf.expand_dims(y_pred, -1)).result(), step=epoch)
#
# # test the model
# # @tf.function
# def test_step(model, images, loss_fun):
#     m_i, m_k, map_att, gtmap, score, y_true, y_pred, n_classes, batch_size = model(images)
#     loss = loss_fun(m_i, m_k, map_att, gtmap, score, y_true, y_pred, n_classes, batch_size)
#     classification = self.classifier_unseen()
#     print("Current TestLoss: {}".format(loss))
#
#
# # testing by running
#
# if __name__ == '__main__':
#     tf.compat.v1.enable_eager_execution()
#     gpu = tf.config.experimental.list_physical_devices('GPU')
#     print("Num GPUs Available: ", len(gpu))
#     if len(gpu) > 0:
#         tf.config.experimental.set_memory_growth(gpu[0], True)
#         tf.config.experimental.set_memory_growth(gpu[1], True)
#
#     # read dataset
#     path_root = os.path.abspath(os.path.dirname(__file__))
#     database = DataSet("/Volumes/Watermelon")#path_root)
#     PHI = database.get_phi()
#     DS, DS_test = database.load_gpu(batch_size=5)  # image_batch, label_batch
#     modelaki = FinalModel()
#
#     # define loss and opt functions
#     loss_fun = Loss().final_loss
#     step = tf.Variable(0, trainable=False)
#     boundaries = [187*5, 187*10]
#     values = [0.05, 0.005, 0.0005]
#     learning_rate_fn = PiecewiseConstantDecay(boundaries, values)
#     # Later, whenever we perform an optimization step, we pass in the step.
#     learning_rate = learning_rate_fn(step)
#     opt_fun = tfa.optimizers.SGDW(learning_rate=learning_rate, weight_decay=5*1e-4, momentum=0.9)
#     #opt_fun = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9)
#
#     # define checkpoint settings
#     ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=opt_fun, net=modelaki)
#     manager = tf.train.CheckpointManager(ckpt, path_root + '/tf_ckpts', max_to_keep=10)  # keep only the three most recent checkpoints
#     ckpt.restore(manager.latest_checkpoint)  # pickup training from where you left off
#
#     # define train and test loss and accuracy
#     train_loss = tf.keras.metrics.Mean(name='train_loss')
#     train_accuracy = tf.keras.metrics.Accuracy(name='train_accuracy')
#     test_loss = tf.keras.metrics.Mean(name='test_loss')
#     test_accuracy = tf.keras.metrics.Accuracy(name='test_accuracy')
#
#     # define data for tensorboard
#     current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
#     train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
#     test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
#     train_summary_writer = tf.summary.create_file_writer(train_log_dir)
#     test_summary_writer = tf.summary.create_file_writer(test_log_dir)
#
#     EPOCHS = 1
#     CHECKEPOCHS = 3
#
#     count = 0
#     # run for each epoch and batch
#     # loss and accuracy are saved every 50 updates
#     # model saved every 3 epochs
#     for epoch in range(EPOCHS):
#         train_loss_results = []
#         train_accuracy_results = []
#         for images, labels in DS:
#             if images.shape[0] == BATCH_SIZE:
#                 train_step(modelaki, images, labels, PHI, loss_fun, opt_fun, epoch)
#                 train_loss_results.append(train_loss.result())
#                 train_accuracy_results.append(train_accuracy.result())
#                 count += 1
#                 if count % 50 == 0:
#                     with open(path_root + '/log.txt', 'a') as temp:
#                         temp.write('Epoch: {}, step: {}, train_Loss: {}, train_Accuracy: {}\n'.format(
#                             epoch + 1, count, sum(train_loss_results) / len(train_accuracy_results),
#                             sum(train_accuracy_results) / len(train_accuracy_results)))
#         #ckpt.step.assign_add(1)
#         #if int(ckpt.step) % CHECKEPOCHS == 0:
#         #    save_path = manager.save()
#
#         template = 'Epoch {}, Loss: {}'
#         print(template.format(epoch + 1, train_loss))
#
#     # TEST UNSEEN CLASSES
#     test_step(modelaki, DS_test, loss_fun)
