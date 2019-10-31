import tensorflow as tf
from tensorflow.keras.optimizers import SGD
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import layers


class ReShape224(layers.Layer):
    """
    This is used to resize full image and attended part-images generated from Cropping subnet
    Args:
        tensor of shape (32,448,448,3) which comes from Cropping subnet
    Returns:
        tensor of shape (dynamic_batch_size, 224,224,3).
        (in order to resize the size of the batch becomes dynamic)
    """

    def __init__(self):
        super(ReShape224, self).__init__()

    def call(self, input):
        out = tf.image.resize(input, [224,224])
        return out



class RCN(tf.keras.Model):
    def __init__(self, hidden_unit, map_size=14, image_size=448):
        super(RCN, self).__init__()
        self.flat = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(hidden_unit, activation="relu")#, input_shape=(map_size * map_size))
        self.fc2 = tf.keras.layers.Dense(3)
        self.activation = tf.keras.layers.ReLU(max_value=map_size-1)#sigmoid
        self.map_size = map_size

    def call(self, input, training=True, k=10):
        out = self.flat(input)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.activation(out)
        #out = tf.dtypes.cast(out, tf.int32)
        mask = self.boxcar(out, k)

        return mask

    def boxcar(self, t, k):
        #V = tf.zeros([t.shape[0], self.map_size, self.map_size])
        V = []
        X = tf.range(start=0, limit=self.map_size)
        Y = tf.range(start=0, limit=self.map_size)
        X, Y = tf.meshgrid(X, Y)
        X = tf.dtypes.cast(X, tf.float32)
        Y = tf.dtypes.cast(Y, tf.float32)
        for n in range(t.shape[0]):
            '''
            for i in range(V.shape[1]):
                for j in range(V.shape[2]):
                    V_x = 1 / (1 + tf.math.exp(-k * (i - t[n, 0] + 0.5 * t[n, 2]))) - 1 / (1 + tf.math.exp(-k * (i - t[n, 0] - 0.5 * t[n, 2])))
                    V_y = 1 / (1 + tf.math.exp(-k * (j - t[n, 1] + 0.5 * t[n, 2]))) - 1 / (1 + tf.math.exp(-k * (j - t[n, 1] - 0.5 * t[n, 2])))
                    V[n, i, j] = tf.multiply(V_x, V_y)
            '''
            V_x = 1 / (tf.math.exp(-k * (tf.math.add((X-t[n,0]), t[n,2]/2))) + 1) - 1 / (1 + tf.math.exp(-k * (tf.math.subtract((X-t[n,0]), t[n,2]/2))))
            V_y = 1 / (1 + tf.math.exp(-k * tf.math.add(Y-t[n,1], t[n,2]/2))) - 1 / (1 + tf.math.exp(-k * tf.math.subtract(Y-t[n,1], t[n,2]/2)))
            #V[n] = tf.multiply(V_x, V_y)
            V.append(tf.multiply(V_x, V_y))

        return tf.stack(V)

class Crop(layers.Layer):

    def __init__(self):
        super(Crop, self).__init__()

    #@tf.function
    def call(self, image, mask):
        crop_image = tf.zeros(image.shape)
        #upsample
        mask = tf.expand_dims(mask, -1)
        mask = tf.broadcast_to(mask, [mask.shape[0], mask.shape[1], mask.shape[2], 3])
        mask = tf.image.resize(mask, size=[image.shape[1], image.shape[2]])
        crop_image = tf.multiply(image, mask)

        return crop_image, mask

