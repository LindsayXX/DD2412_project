import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import tensorflow as tf
from Multi_attention_subnet import Kmeans, WeightedSum, Average_Pooling
from Cropping_subnet import RCN, Crop
from dataloader import DataSet
import os
from tensorflow.python.keras import backend as K
from Losses import Loss

CHANNELS = 512
N_CLASSES = 200
SEMANTIC_SIZE = 28
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
        self.kmeans = Kmeans(clusters_n=2, iterations=10)
        self.average_pooling_0 = Average_Pooling()
        self.average_pooling_1 = Average_Pooling()

        self.fc1_1 = tf.keras.layers.Dense(512, activation="relu")
        self.fc1_2 = tf.keras.layers.Dense(512, activation="relu")

        self.fc2_1 = tf.keras.layers.Dense(512, activation="sigmoid")
        self.fc2_2 = tf.keras.layers.Dense(512, activation="sigmoid")

        self.bn0 = tf.keras.layers.BatchNormalization()
        self.bn1 = tf.keras.layers.BatchNormalization()

        self.weighted_sum0 = WeightedSum()
        self.weighted_sum1 = WeightedSum()

        # cropping net
        self.crop_net0 = RCN(hidden_unit=14, map_size=14, image_size=448)
        self.crop_net1 = RCN(hidden_unit=14, map_size=14, image_size=448)
        self.crop0 = Crop()
        self.crop1 = Crop()

    def call(self, x):
        # MULTI ATTENTION SUBNET
        # x will be image_batch of shape (BATCH,448,448,3)
        # print("VGG")
        feature_map = self.vgg_features_initial(x)  # gives an output of shape (BATCH,14,14,512)
        # print("KMEANS")
        batch_cluster0, batch_cluster1 = self.kmeans(
            feature_map)  # gives two lists containing tensors of shape (512,14,14)
        # print("AVR POOL")
        # print(batch_cluster0.shape, batch_cluster1.shape)
        p1 = self.average_pooling_0(
            batch_cluster0)  # gives a list of length=batch_size containing tensors of shape (512,)
        p2 = self.average_pooling_1(
            batch_cluster1)  # gives a list of length=batch_size containing tensors of shape (512,)
        # print("FC")

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

        # print("WS")
        m0 = self.weighted_sum0(feature_map, a0)  # gives tensor of shape (BATCH,14,14)
        m1 = self.weighted_sum1(feature_map, a1)  # gives tensor of shape (BATCH,14,14)

        # m0 = tf.convert_to_tensor(np.load("im0.npy"))
        # m1 = tf.convert_to_tensor(np.load("im1.npy"))
        attmap_out = tf.TensorArray(tf.float32, 2)
        attmap_out.write(0, m0)
        attmap_out.write(1, m1)
        attmap_out = attmap_out.stack()

        # print("CROP")
        # CROPPING SUBNET
        mask0 = self.crop_net0(m0)  # shape(BATCH,14,14)
        mask1 = self.crop_net1(m1)  # shape(BATCH,14,14)
        croped0, newmask0 = self.crop0(x, mask0)  # of shape (BATCH,448,448,3)
        croped1, newmask1 = self.crop1(x, mask1)  # of shape (BATCH,448,448,3)
        # attention_crop_out = tf.TensorArray(tf.float32, size=2)
        # attention_crop_out.write(0, mask0)
        # attention_crop_out.write(1, mask1)
        # attention_crop_out = attention_crop_out.stack()
        # gives 3 tensors of size (batch,448,448,3)

        return m0, m1, mask0, mask1, croped0, croped1, newmask0, newmask1


@tf.function
def train_step(model, image_batch, loss_fun, opt_fun, batch_size):
    with tf.GradientTape() as tape:
        m0, m1, mask0, mask1, croped0, croped1, newmask0, newmask1 = model(image_batch)
        loss = loss_fun(m0, m1, mask0, mask1, batch_size)
    gradients = tape.gradient(loss, model.trainable_variables)
    opt_fun.apply_gradients(zip(gradients, model.trainable_variables))
    #print("Current TrainLoss: {}".format(loss))
    train_loss(loss)
    #train_accuracy(tf.expand_dims(labels, -1), tf.expand_dims(y_pred, -1))
    return croped0, newmask0


'''
# test the model
# @tf.function
def test_step(model, images, loss_fun):
    m_i, m_k, map_att, gtmap, score, y_true, y_pred, n_classes, batch_size = model(images)
    loss = loss_fun(m_i, m_k, map_att, gtmap, score, y_true, y_pred, n_classes, batch_size)
    print("Current TestLoss: {}".format(loss))
'''

# testing by running

if __name__ == '__main__':

    # tf.compat.v1.enable_eager_execution()
    gpu = tf.config.experimental.list_physical_devices('GPU')
    print("Num GPUs Available: ", len(gpu))
    # path_root = os.path.abspath(os.path.dirname(__file__))
    database = DataSet('../basemodel') #"/content/gdrive/My Drive/data")  # path_root)

    # DS, DS_test = database.load_gpu()  # image_batch, label_batch
    DS = database.load(GPU=False, train=True, batch_size=4)
    # DS_test = database.load(GPU=False, train=False, batch_size = 32)

    modelaki = FinalModel()

    loss_fun = Loss().loss_MA
    opt_fun = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9)

    # ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=opt_fun, net=modelaki)
    # manager = tf.train.CheckpointManager(ckpt, path_root + '/tf_ckpts',
    #                                     max_to_keep=3)  # keep only the three most recent checkpoints
    # ckpt.restore(manager.latest_checkpoint)  # pickup training from where you left off

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.Accuracy(name='train_accuracy')
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.Accuracy(name='test_accuracy')

    # m0, m1 = modelaki.call(image_batch)

    EPOCHS = 5
    CHECKEPOCHS = 1

    # train_loss = tf.keras.metrics.Mean(name='train_loss')
    # image_batch, label_batch = next(iter(DS))
    for epoch in range(EPOCHS):
        train_loss_results = []
        train_accuracy_results = []
        for images, labels in DS:
            #if images.shape[0] == 4:
            #m0, m1, mask0, mask1, croped0, croped1, newmask0, newmask1 = modelaki(images)
            croped0, newmask0 = train_step(modelaki, images, loss_fun, opt_fun, batch_size=4)
            tf.print(croped0.shape, newmask0.shape)
            plt.imshow(K.eval(croped0)[0])#.astype(np.float32) #, interpolation='nearest')
            plt.show()
            plt.imshow(images[0].numpy().astype(np.float32))
            plt.show()
            #plt.imshow(K.eval(newmask0)[0].astype(np.uint8))
            plt.imshow(K.eval(newmask0)[0])
            plt.show()

            tf.print('Epoch {}, train_Loss: {}\n'.format(epoch + 1, train_loss.result()))
            # train_loss_results.append(train_loss.result())
            # train_accuracy_results.append(train_accuracy.result())

        # print('Epoch {}, Loss: {}'.format(epoch + 1, train_loss.result()))
