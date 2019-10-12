import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D, Activation, GlobalAveragePooling2D
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD



IMG_SIZE = 448
'''Multi-Attention Subnet'''
#TODO: continue the network

# feature extraction/representation
'''
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
base_model = tf.keras.applications.VGG19(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')
# freeze the convolutional layers(fix the weight)
base_model.trainable = False
feature_batch = base_model(image_batch)
print(feature_batch.shape)
model = Sequential([base_model])
sgd = SGD(lr=0.05, decay=5 * e-4, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])
global_average_layer = GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)
'''



'''Region Cropping Subnet'''
#TODO: need rearrange the code to different files
def RCN(input):
    '''
    :param input: attention map with size： BATCH_SIZE * IMG_SIZE * IMG_SIZE * 1
    :return: region cropping subnet
    '''
    layers = [Dense(#TODO: HIDDEN_SIZE, input_dim=(IMG_SIZE, IMG_SIZE)),
            Activation('relu'),
            Dense(3,)]
    model = Sequential(layers)
    # should we compile here?

    return model

# boxcar mask
def f(x, k=10):
    return 1/(1 + tf.math.exp(-k * x))

def mask(image, coord):
    '''
    :param image: IMG_SIZE * IMG_SIZE *3
    :param coord: [tx, ty, ts], type?
    :return: croped image, IMG_SIZE/2 * IMG_SIZE/2 * 3
    '''
    # TODO: need correct
    v_x = f(image[:, 1] - coord[0] + 0.5 * coord[2]) - f(image[:, 1] - coord[0] - 0.5 * coord[2])
    v_y = f(image[1, :] - coord[1] + 0.5 * coord[2]) - f(image[1, :] - coord[1] - 0.5 * coord[2])
    V = tf.matmul(v_x, v_y)
    X_part= tf.math.multiply(image, V)

    return X_part



