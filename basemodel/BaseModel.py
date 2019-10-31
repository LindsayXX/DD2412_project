from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Activation, GlobalAveragePooling2D
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
import matplotlib.pyplot as plt
import numpy as np
from Multi_attention_subnet import VGG_feature, Kmeans, Average_Pooling, Fc, WeightedSum, Average_Pooling_basemodel
from Cropping_subnet import ReShape224
from Joint_feature_learning_subnet import Scores
import tensorflow.keras.optimizers as opt
from dataloader import DataSet
from Losses import Loss
import os


# IMG_SIZE = 448
CHANNELS = 512
BATCH_SIZE = 32


class BaseModel(Model):

    def __init__(self, n_class):
        super(BaseModel, self).__init__()
        self.reshape224 = ReShape224()
        self.vgg_features = VGG_feature()
        self.average_pooling = Average_Pooling_basemodel()
        self.score = Scores()
        #self.fc = Fc(CHANNELS)
        self.fc = Dense(n_class, activation="softmax")#, input_shape=(1,28))

    def call(self, x, phi):
        resized = self.reshape224(x)  # δίνει (32,224,224,3)
        features = self.vgg_features(resized)  # δίνει (32,7,7,512)
        global_theta = self.average_pooling(features)  # tensor of shape (32,512)
        global_scores, out = self.score(global_theta, phi)
        out = self.fc(out)  #prediction (batch_size, 200)

        return global_scores, out


# test the model
@tf.function
def test_step(model, images):
    scores = model(images)
    loss = Loss().loss_CLS(scores)
    print("Current TestLoss: {}".format(loss))

#@tf.function
def train_step(model, image_batch, loss_fun, opt_fun):
    with tf.GradientTape() as tape:
        scores, outs = model(image_batch)
        loss = loss_fun(scores)
    gradients = tape.gradient(loss, model.trainable_variables)  # ti einai trainable variables?
    print(gradients)
    opt_fun.apply_gradients(zip(gradients, model.trainable_variables))
    #print("Current TrainLoss: {}".format(loss))
    return loss

if __name__ == '__main__':
    path_root = os.path.abspath(os.path.dirname(__file__))
    bird_data = DataSet(path_root)
    train_ds, phi = bird_data.load(GPU=False, train=True, batch_size=32)
    test_ds, useless_semantic = bird_data.load(GPU=False, train=False, batch_size=32)
    image_batch, label_batch = next(iter(train_ds))

    basemodel = BaseModel(200)

    global_scores, out = basemodel(image_batch)#, phi)
    # sizes (32,1),(1,200)

    EPOCHS = 5

    #train_loss = tf.keras.metrics.Mean(name='train_loss')

    for epoch in range(EPOCHS):

        loss_fun = Loss().loss_CLS
        opt_fun = opt.Adam()
        #for images, labels in zip(image_batch, label_batch):
        train_loss = train_step(basemodel, image_batch, label_batch, loss_fun, opt_fun)

        # for test_images, test_labels in test_ds:
        #    test_step(test_images, test_labels)

        template = 'Epoch {}, Loss: {}'
        print(template.format(epoch + 1, train_loss))

        # Reset the metrics for the next epoch
        # train_loss.reset_states()
        #train_accuracy.reset_states()
        #test_loss.reset_states()
        #test_accuracy.reset_states()

# TODO: pre-train vgg only on birds
# TODO: global variables of this module may differ from the subnetworks module.
