import tensorflow as tf
from tensorflow.keras.optimizers import SGD
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import layers


class ReShape224(layers.Layer):

    def __init__(self,input):
        super(reshapes,self).__init__()
        self.input = input

    def call(self, input):
        out = tf.reshape(self.input, [-1,224,224,3])
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
        super(Crop,self).__init__()

    #@tf.function
    def call(self, image, mask):
        crop_image = tf.zeros(image.shape)
        #upsample
        mask = tf.expand_dims(mask, -1)
        mask = tf.broadcast_to(mask, [mask.shape[0], mask.shape[1], mask.shape[2], 3])
        mask = tf.image.resize(mask, size=[image.shape[1], image.shape[2]])
        tf.multiply(image, mask)

        return crop_image, mask


if __name__ == '__main__':
    # test
    batch_size = 3
    sample_input = tf.random.normal([batch_size, 14, 14])

    crop_net = RCN(hidden_unit=14, map_size=14, image_size=448)
    mask = crop_net.call(sample_input, k=10)  # of shape (3,14,14)
    sample_image = tf.random.normal([batch_size, 448, 448, 3])
    crop_image, newmask = crop(sample_image, mask)


    for i in range(batch_size):
        plt.subplot(131)
        plt.imshow(sample_image[i].numpy())
        plt.subplot(132)
        plt.imshow(newmask[i].numpy())
        plt.subplot(133)
        #plt.imshow(tf.multiply(sample_image[i], tf.ones(sample_image.shape))[0].numpy())
        plt.imshow(crop_image[i].numpy())
        plt.show()
