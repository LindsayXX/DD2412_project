import tensorflow as tf
from tensorflow_core.python.keras.optimizer_v2.learning_rate_schedule import PiecewiseConstantDecay

from dataloader import DataSet
from final_model.Losses import Loss
from final_model.FinalModel import FinalModel
import tensorflow_addons as tfa

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
    database = DataSet("/Volumes/Watermelon")
    PHI = database.get_phi()
    ds = database.load(batch_size=5)
    for im, label in ds:
    #im_path = "/Volumes/Watermelon/CUB_200_2011/CUB_200_2011/images/059.California_Gull/"
    #img = tf.io.read_file(im_path)
    #im = database.decode_img(img)
        m0, m1, mask0, mask1, global_scores, local_scores0, local_scores1, \
        global_phi, local0_phi, local1_phi, y_pred, C = net(im, PHI) #tf.expand_dims(im,0)
    print("wena")
