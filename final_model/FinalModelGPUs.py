# This code was adapted in order to run in multiple GPUs

from __future__ import absolute_import, division, print_function, unicode_literals

import datetime
import os
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import Model
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay
from Classification import Classifier
from Multi_attention_subnet import VGG_feature, Kmeans, Average_Pooling, Fc, WeightedSum
from Cropping_subnet import ReShape224, RCN, Crop
from Joint_feature_learning_subnet import Scores
from dataloader import DataSet
from Losses import Loss
import sys
sys.path.append("../src")
from jointmodel import JFL

CHANNELS = 512
N_CLASSES = 200
SEMANTIC_SIZE = 28
BATCH_SIZE = 5
IMG_SIZE = 448
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

# read dataset
database = DataSet("/Volumes/Watermelon")
PHI = database.get_phi()
DS, DS_test = database.load_gpu(batch_size=BATCH_SIZE)

tf.compat.v1.enable_eager_execution()
strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

BUFFER_SIZE = 5
BATCH_SIZE_PER_REPLICA = 32
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
EPOCHS = 30

train_dataset = DS
train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)


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
        # VGG
        feature_map = self.vgg_features_initial(x)  # gives an output of shape (BATCH,14,14,512)
        # KMEANS
        # gives two lists containing tensors of shape (512,14,14)
        batch_cluster0, batch_cluster1 = self.kmeans(feature_map)
        # AVR POOL
        # gives a list of length=batch_size containing tensors of shape (512,)
        p1 = self.average_pooling_0(batch_cluster0)
        p2 = self.average_pooling_1(batch_cluster1)
        # FC

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

        # WS
        m0 = self.weighted_sum0(feature_map, a0)  # gives tensor of shape (BATCH,14,14)
        m1 = self.weighted_sum1(feature_map, a1)  # gives tensor of shape (BATCH,14,14)

        # CROPPING SUBNET
        mask0 = self.crop_net0(m0)  # shape(BATCH,14,14)
        mask1 = self.crop_net1(m1)  # shape(BATCH,14,14)
        croped0, newmask1 = self.crop0(x, mask0)  # of shape (BATCH,448,448,3)
        croped1, newmask2 = self.crop1(x, mask1)  # of shape (BATCH,448,448,3)

        # RESHAPE
        # JOINT FEATURE LEARNING SUBNE
        # resizing the outputs of cropping network
        full_image = self.reshape_global(x)
        attended_part0 = self.reshape_local0(croped0)
        attended_part1 = self.reshape_local1(croped1)

        # VGG
        # feeding the 3 images into VGG nets
        full_image = self.vgg_features_global(full_image)
        attended_part0 = self.vgg_features_local0(attended_part0)
        attended_part1 = self.vgg_features_local1(attended_part1)

        # THETAS
        # creating thetas
        global_theta = self.average_pooling_global(full_image)
        local_theta0 = self.average_pooling_local0(attended_part0)
        local_theta1 = self.average_pooling_local1(attended_part1)

        # computing the scores
        # Computing Scores...
        global_scores, global_phi = self.joint_net_global.call(global_theta,
                                                               phi)  # self.global_score(global_theta, phi)
        local_scores0, local0_phi = self.joint_net_local0.call(local_theta0, phi)  # self.score0(local_theta0, phi)
        local_scores1, local1_phi = self.joint_net_local1.call(local_theta1, phi)  # self.score1(local_theta1, phi)

        # print(scores_out.shape)
        y_pred = self.classifier(global_scores, local_scores0, local_scores1)

        return m0, m1, mask0, mask1, global_scores, local_scores0, local_scores1, \
               global_phi, local0_phi, local1_phi, y_pred, self.C

def create_model():
    return FinalModel()

# Create a checkpoint directory to store the checkpoints.
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

with strategy.scope():
    loss_fun = Loss().final_loss
    def compute_loss(m0, m1, mask0, mask1, global_scores, local_scores0, local_scores1, global_phi,
                                local0_phi, local1_phi, labels, y_pred, N_CLASSES, BATCH_SIZE, C):
        loss = loss_fun(m0, m1, mask0, mask1, global_scores, local_scores0, local_scores1, global_phi, local0_phi,
                        local1_phi, labels, y_pred, N_CLASSES, BATCH_SIZE, C)
        # Compute loss that is scaled by sample_weight and by global batch size.
        loss_avg = tf.nn.compute_average_loss(loss, global_batch_size=GLOBAL_BATCH_SIZE)
        return loss_avg

with strategy.scope():
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.Accuracy(name='train_accuracy')

with strategy.scope():
    model = create_model()
    step = tf.Variable(0, trainable=False)
    boundaries = [187 * 5, 187 * 10]
    values = [0.05, 0.005, 0.0005]
    learning_rate_fn = PiecewiseConstantDecay(boundaries, values)
    # Later, whenever we perform an optimization step, we pass in the step.
    learning_rate = learning_rate_fn(step)
    optimizer = tfa.optimizers.SGDW(learning_rate=learning_rate, weight_decay=5 * 1e-4, momentum=0.9)
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)

with strategy.scope():
    def train_step(inputs):
        images, labels, phi = inputs
        with tf.GradientTape() as tape:
            m0, m1, mask0, mask1, global_scores, local_scores0, local_scores1, global_phi, local0_phi, \
            local1_phi, y_pred, C = model(images, phi, training=True)
            loss = compute_loss(m0, m1, mask0, mask1, global_scores, local_scores0, local_scores1, global_phi,
                                local0_phi, local1_phi, labels, y_pred, N_CLASSES, BATCH_SIZE, C)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_accuracy.update_state(labels, y_pred)
        return loss

with strategy.scope():
    #@tf.function
    def distributed_train_step(dataset_inputs):
        per_replica_losses = strategy.experimental_run_v2(train_step, args=(dataset_inputs,))
        return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

    for epoch in range(EPOCHS):
        # TRAIN LOOP
        total_loss = 0.0
        num_batches = 0
        for x in train_dist_dataset:
            image, label = x
            total_loss += distributed_train_step((image, label, PHI))
            num_batches += 1
        train_loss = total_loss / num_batches

        if epoch % 2 == 0:
            checkpoint.save(checkpoint_prefix)

        template = ("Epoch {}, Loss: {}, Accuracy: {}")
        print(template.format(epoch + 1, train_loss, train_accuracy.result() * 100))

        train_accuracy.reset_states()
