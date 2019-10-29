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


#IMG_SIZE = 448
CHANNELS = 512
BATCH_SIZE = 32


class FinalModel(Model):


    def __init__(self):
        super(FinalModel, self).__init__()

        # MULTI ATTENTION SUBNET
        self.vgg_features = VGG_feature()
        self.kmeans = Kmeans(clusters_n=2, iterations = 10)
        self.average_pooling = Average_Pooling()
        self.fc = Fc(CHANNELS)
        self.weighted_sum = WeightedSum()

        '''
        # cropping net
        self.crop_net = RCN(hidden_unit=14, map_size=14, image_size=448)
        self.crop = Crop()
        '''
        self.reshape224 = ReShape224()
    
    
        # joint feature learning subnet
        self.score = Scores()

    def call(self,x):

        # MULTI ATTENTION SUBNET
        # x will be image_batch of shape (BATCH,448,448,3)
        feature_map = self.vgg_features(x) # gives an output of shape (BATCH,14,14,512)
        batch_cluster0, batch_cluster1 = self.kmeans(feature_map) # gives two lists containing tensors of shape (512,14,14)

        p1 = self.average_pooling(batch_cluster0) # gives a list of length=batch_size containing tensors of shape (512,)
        p2 = self.average_pooling(batch_cluster1) # gives a list of length=batch_size containing tensors of shape (512,)

        a0 = self.fc(p1) # gives tensor of shape (BATCH,512)
        a1 = self.fc(p2) # gives tensor of shape (BATCH,512)

        m0 = self.weighted_sum(feature_map, a0) # gives tensor of shape (BATCH,14,14)
        m1 = self.weighted_sum(feature_map, a1) # gives tensor of shape (BATCH,14,14)

        '''
        #CROPPING SUBNET
        mask1 = self.crop_net(m0) #shape(BATCH,14,14)
        mask2 = self.crop_net(m1) #shape(BATCH,14,14)
        croped0, newmask1 = self.crop(x,mask1) #of shape (BATCH,448,448,3)
        croped1, newmask2 = self.crop(x,mask2) #of shape (BATCH,448,448,3)
        gives 3 tensors of size (batch,448,448,3)
        '''

        #JOINT FEATURE LEARNING SUBNET
        #resizing the outputs of cropping network
        full_image = ReShape224(x, [-1,224,224,3])
        attended_part0 = ReShape224(croped0, [-1,224,224,3])
        attended_part1 = ReShape224(croped1, [-1,224,224,3])
        #from now on the batch size is dynamic!!

        #feeding the 3 images into VGG nets
        full_image     = self.vgg_features(full_image)
        attended_part0 = self.vgg_features(attended_part0)
        attended_part1 = self.vgg_features(attended_part1)

        #creating thetas
        global_theta = self.average_pooling(full_image)
        local_theta0 = self.average_pooling(attended_part0)
        local_theta1 = self.average_pooling(attended_part1)

        #computing the scores
        global_scores = self.score(global_theta)
        local_scores0 = self.score(local_theta0)
        local_scores1 = self.score(local_theta1)









        #













        return m0,m1



#testing by running

if __name__ == '__main__':
    database = Database()
    image_batch, label_batch = database.call()
    #image batch is of shape(32,448,448,3) and label_batch is(32,200)
    modelaki = FinalModel()
    m0,m1 = modelaki.call(image_batch)
    



# TODO: pre-train vgg only on birds
# TODO: global variables of this module may differ from the subnetworks module.
















