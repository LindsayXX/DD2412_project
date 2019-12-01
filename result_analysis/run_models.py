import os

import tensorflow as tf
from tensorflow_core.python.keras.optimizer_v2.learning_rate_schedule import PiecewiseConstantDecay

from Classification import Classifier_Unseen
from dataloader import DataSet
from final_model.Losses import Loss
from final_model.FinalModel import FinalModel
import tensorflow_addons as tfa

def test_step(model, image_batch, W, seen_classes, unseen_classes):
    m0, m1, mask0, mask1, scores, \
    phi, y_pred, C = model(image_batch, PHI)
    class_unseen = Classifier_Unseen(W, C)
    classification = class_unseen(phi, scores)
    #print("Current TestLoss: {}".format(loss))
    return classification

if __name__ == '__main__':
    # LOSS AND OPT
    loss_fun = Loss().final_loss
    step = tf.Variable(0, trainable=False)
    boundaries = [187 * 5, 187 * 10]
    values = [0.05, 0.005, 0.0005]
    learning_rate_fn = PiecewiseConstantDecay(boundaries, values)
    # Later, whenever we perform an optimization step, we pass in the step.
    learning_rate = learning_rate_fn(step)
    opt = tfa.optimizers.SGDW(learning_rate=learning_rate, weight_decay=5 * 1e-4, momentum=0.9)

    # MODEL
    #net = FinalModel()
    #new_root = tf.train.Checkpoint(net=net)
    #status = new_root.restore(tf.train.latest_checkpoint('./tf_ckpts/'))
    net = FinalModel()
    ckpt = tf.train.Checkpoint(step=tf.Variable(1, dtype=tf.int32), optimizer=opt, net=net)
    ckpt.restore(tf.train.latest_checkpoint('./tf_ckpts/'))

    #DATA
    path_root = os.path.abspath(os.path.dirname(__file__))
    bird_data = DataSet("/Volumes/Watermelon")  # DataSet(path_root)
    phi_train = bird_data.get_phi(set=0)
    w = bird_data.get_w(alpha=1)  # (50*150)
    train_class_list, test_class_list = bird_data.get_class_split(mode="easy")
    train_ds = bird_data.load(GPU=False, train=True, batch_size=32)
    #test_ds = bird_data.load(GPU=False, train=False, batch_size=4) #.load_gpu(batch_size=4)
    PHI = bird_data.get_phi(set=0)
    for im, label in train_ds:
        #im_path = "/Volumes/Watermelon/CUB_200_2011/CUB_200_2011/images/059.California_Gull/"
        #img = tf.io.read_file(im_path)
        #im = database.decode_img(img)
        m0, m1, mask0, mask1, scores, phi, y_pred, C = net(im, PHI) #tf.expand_dims(im,0)

    nu = 50
    ns = 150
    W = tf.ones((nu, ns))
    seen_classes = tf.ones((ns, 28))
    unseen_classes = tf.ones((nu, 28))
    phi_test = bird_data.get_phi(set=1)
    for im, label in train_ds:
        classification = test_step(net, im, W, phi_test)

    print("wena")
