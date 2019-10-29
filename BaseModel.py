from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Activation, GlobalAveragePooling2D
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
import matplotlib.pyplot as plt
import numpy as np
from Multi_attention_subnet import VGG_feature, Kmeans, Average_Pooling, Fc, WeightedSum, Average_Pooling_basemodel
from Cropping_subnet import ReShape224, RCN, Crop
from Joint_feature_learning_subnet import Scores
from DataBase import Database

# IMG_SIZE = 448
CHANNELS = 512
BATCH_SIZE = 32


class BaseModel(Model):

    def __init__(self):
        super(BaseModel, self).__init__()

        self.reshape224 = ReShape224()
        self.vgg_features = VGG_feature()
        self.average_pooling = Average_Pooling_basemodel()
        self.score = Scores()
        #self.fc = Fc(CHANNELS)


    def call(self, x):

        resized = self.reshape224(x) #(32,224,224,3)
        features = self.vgg_features(resized) #(32,7,7,512)
        global_theta = self.average_pooling(features) #list of 128 tensors of shape (512,)
        global_scores = self.score(global_theta)

        return global_scores

if __name__ == '__main__':
    database = Database()
    image_batch, label_batch = database.call()  # image batch is of shape(32,448,448,3) and label_batch is(32,200)

    basemodel = BaseModel()

    global_scores = basemodel(image_batch)

    #it works :)


# TODO: pre-train vgg only on birds
# TODO: global variables of this module may differ from the subnetworks module.
















