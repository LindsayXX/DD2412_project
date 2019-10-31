from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Activation, GlobalAveragePooling2D
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
import matplotlib.pyplot as plt
import numpy as np
from Multi_attention_subnet import VGG_feature,Kmeans, Average_Pooling, Fc, WeightedSum
from Cropping_subnet import ReShape224, RCN, Crop
from Joint_feature_learning_subnet import Scores
from DataBase import Database
import tensorflow.keras.optimizers as opt

from Losses import Loss


#IMG_SIZE = 448
CHANNELS = 512
BATCH_SIZE = 32


class FinalModel(Model):

    def __init__(self):
        super(FinalModel, self).__init__()

        # MULTI ATTENTION SUBNET
        self.vgg_features_initial = VGG_feature()
        self.kmeans = Kmeans(clusters_n=2, iterations=10)
        self.average_pooling_0 = Average_Pooling()
        self.average_pooling_1 = Average_Pooling()
        self.fc_0 = Fc(CHANNELS)
        self.fc_1 = Fc(CHANNELS)
        self.weighted_sum0 = WeightedSum()
        self.weighted_sum1 = WeightedSum()


        #cropping net
        self.crop_net0 = RCN(hidden_unit=14, map_size=14, image_size=448)
        self.crop_net1 = RCN(hidden_unit=14, map_size=14, image_size=448)
        self.crop0 = Crop()
        self.crop1 = Crop()

        # joint feature learning subnet
        self.reshape_global = ReShape224()
        self.reshape_local0 = ReShape224()
        self.reshape_local1 = ReShape224()

        self.vgg_features_global = VGG_feature()
        self.vgg_features_local0 = VGG_feature()
        self.vgg_features_local1 = VGG_feature()

        self.average_pooling_global = Average_Pooling()
        self.average_pooling_local0 = Average_Pooling()
        self.average_pooling_local1 = Average_Pooling()

        self.global_score = Scores()
        self.score0 = Scores()
        self.score1 = Scores()

    def call(self, x):

        # # MULTI ATTENTION SUBNET
        # # x will be image_batch of shape (BATCH,448,448,3)
        # print("VGG")
        # feature_map = self.vgg_features_initial(x)  # gives an output of shape (BATCH,14,14,512)
        # print(feature_map.shape)
        # print("KMEANS")
        # batch_cluster0, batch_cluster1 = self.kmeans(feature_map) # gives two lists containing tensors of shape (512,14,14)
        # print("AVR POOL")
        # p1 = self.average_pooling_0(batch_cluster0)  # gives a list of length=batch_size containing tensors of shape (512,)
        # p2 = self.average_pooling_1(batch_cluster1)  # gives a list of length=batch_size containing tensors of shape (512,)
        # print("FC")
        # a0 = self.fc_0(p1)  # gives tensor of shape (BATCH,512)
        # a1 = self.fc_1(p2)  # gives tensor of shape (BATCH,512)
        # print("WS")
        # m0 = self.weighted_sum0(feature_map, a0)  # gives tensor of shape (BATCH,14,14)
        # m1 = self.weighted_sum1(feature_map, a1)  # gives tensor of shape (BATCH,14,14)

        m0 = tf.convert_to_tensor(np.load("im0.npy"))
        m1 = tf.convert_to_tensor(np.load("im1.npy"))
        print("CROP")
        #CROPPING SUBNET
        mask1 = self.crop_net0(m0) #shape(BATCH,14,14)
        mask2 = self.crop_net1(m1) #shape(BATCH,14,14)
        croped0, newmask1 = self.crop0(x, mask1) #of shape (BATCH,448,448,3)
        croped1, newmask2 = self.crop1(x, mask2) #of shape (BATCH,448,448,3)
        #gives 3 tensors of size (batch,448,448,3)

        print("RESHAPE")
        #JOINT FEATURE LEARNING SUBNE
        #resizing the outputs of cropping network
        full_image = self.reshape_global(x)
        attended_part0 = self.reshape_local0(croped0)
        attended_part1 = self.reshape_local1(croped1)

        print("VGG")
        #feeding the 3 images into VGG nets
        full_image     = self.vgg_features_global(full_image)
        attended_part0 = self.vgg_features_local0(attended_part0)
        attended_part1 = self.vgg_features_local1(attended_part1)

        print("THETAS")
        #creating thetas
        global_theta = self.average_pooling_global(full_image)
        local_theta0 = self.average_pooling_local0(attended_part0)
        local_theta1 = self.average_pooling_local1(attended_part1)

        #computing the scores
        phi = None
        global_scores = self.global_score(global_theta, phi)
        local_scores0 = self.score0(local_theta0, phi)
        local_scores1 = self.score1(local_theta1, phi)

        return m0, m1, [m0, m1], [croped0, croped1], [None , None, None], None, None, 200, 32


@tf.function
def train_step(model, image_batch, loss_fun, opt_fun):
    with tf.GradientTape() as tape:
        m_i, m_k, map_att, gtmap, score, y_true, y_pred, n_classes, batch_size = model(image_batch)
        loss = loss_fun(m_i, m_k, map_att, gtmap, score, y_true, y_pred, n_classes, batch_size)
    gradients = tape.gradient(loss, model.trainable_variables)
    opt_fun.apply_gradients(zip(gradients, model.trainable_variables))
    print("Current TrainLoss: {}".format(loss))
    return loss

# test the model
#@tf.function
def test_step(model, images, loss_fun):
    m_i, m_k, map_att, gtmap, score, y_true, y_pred, n_classes, batch_size = model(images)
    loss = loss_fun(m_i, m_k, map_att, gtmap, score, y_true, y_pred, n_classes, batch_size)
    print("Current TestLoss: {}".format(loss))


#testing by running

if __name__ == '__main__':
    database = Database()
    image_batch, label_batch = database.call()
    #image batch is of shape(32,448,448,3) and label_batch is(32,200)
    modelaki = FinalModel()
    #m0, m1 = modelaki.call(image_batch)
    
    EPOCHS = 5

    #train_loss = tf.keras.metrics.Mean(name='train_loss')

    for epoch in range(EPOCHS):
        loss_fun = Loss().final_loss
        opt_fun = opt.Adam()
        #for images, labels in zip(image_batch, label_batch):
        train_loss = train_step(modelaki, image_batch, loss_fun, opt_fun)

        # for test_images, test_labels in test_ds:
        #    test_step(test_images, test_labels)

        template = 'Epoch {}, Loss: {}'
        print(template.format(epoch + 1, train_loss))



# TODO: pre-train vgg only on birds
# TODO: global variables of this module may differ from the subnetworks module.
















