import tensorflow as tf
from tensorflow.keras.optimizers import SGD
import matplotlib.pyplot as plt
import numpy as np

class RCN(tf.keras.Model):
    def __init__(self, hidden_unit, map_size=14):
        super(RCN, self).__init__()
        self.flat = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(hidden_unit)#, input_shape=(map_size * map_size))
        self.fc2 = tf.keras.layers.Dense(3)
        self.map_size = map_size

    def call(self, input, training=True):
        out = self.flat(input)
        out = self.fc1(input)
        out = self.fc2(out)
        #out = tf.dtypes.cast(out, tf.int32)
        mask = self.boxcar(out)

        return mask

    def boxcar(self, t, k=10):
        V = tf.zeros([t.shape[0], self.map_size, self.map_size])
        for k in range(t.shape[0]):
            for i in range(V.shape[1]):
                for j in range(V.shape[2]):
                    V_x = 1 / (1 + tf.math.exp(-k * (i - t[k, 0] + 0.5 * t[k, 2]))) - 1 / (1 + tf.math.exp(-k * (i - t[k, 2] - 0.5 * t[k, 3])))
                    V_y = 1 / (1 + tf.math.exp(-k * (j - t[k, 1] + 0.5 * t[k, 2]))) - 1 / (1 + tf.math.exp(-k * (j - t[k, 1] - 0.5 * t[k, 2])))
                    V[k, i, j] = V_x * V_y

        return V

#@tf.function
def loss_CPT(map, gtmap, batch_size=32):
    diff = tf.math.abs(map, gtmap)
    return tf.nn.l2_loss(diff)/batch_size

def crop(image, mask, batch_size=32):
    # upsample
    for c in range(mask.shape[0]):
        mask[c] = tf.image.resize_images(mask[c], [image.shape[1], image.shape[2]])
    crop_image = tf.zeros(image.shape)
    for i in range(image.shape[0]):#batch
        for j in range(image.shape[-1]):#channel
            crop_image[i, :, :, j] = tf.math.multiply(image[i, :, :, j], mask)

    return crop_image


if __name__ == '__main__':
    # test
    batch_size =  1
    sample_input = tf.random.normal([batch_size, 14, 14])
    crop_net = RCN(hidden_unit=14, map_size=14)
    mask = crop_net.call(sample_input)
    sample_image = tf.random.normal([batch_size, 448, 448, 3])
    crop_image = crop(sample_image, mask)
    for i in range(batch_size):
        plt.imshow(sample_image[i].numpy())
        plt.imshow(crop_image[i].numpy())
        plt.show()
