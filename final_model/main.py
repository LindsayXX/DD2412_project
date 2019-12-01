import os
import tensorflow as tf
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay
import tensorflow_addons as tfa
import datetime

import FinalModel
from Classification import Classifier_Unseen
from Losses import Loss
from dataloader import DataSet

import sys
sys.path.append("../src")
from jointmodel import JFL

CHANNELS = 512
BATCH_SIZE = 32
N_CLASSES = 200
SEMANTIC_SIZE = 28
IMG_SIZE = 448
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

@tf.function
def train_step(model, image_batch, y_true, PHI, loss_fun, opt_fun, epoch):
    with tf.GradientTape() as tape:
        m0, m1, mask0, mask1, scores, phi, y_pred, C = model(image_batch, PHI)
        loss = loss_fun(m0, m1, mask0, mask1, scores, phi, y_true, image_batch.shape[0], C)
    gradients = tape.gradient(loss, model.trainable_variables)
    opt_fun.apply_gradients(zip(gradients, model.trainable_variables))

    # summary for tensorboard
    with train_summary_writer.as_default():
        tf.summary.scalar('loss', loss.result(), step=epoch)
        tf.summary.scalar('accuracy', train_accuracy(tf.expand_dims(y_true, -1),
                                                     tf.expand_dims(y_pred, -1)).result(), step=epoch)

# test the model
# @tf.function
def test_step(model, image_batch, W, phi_test):
    m0, m1, mask0, mask1, scores, phi, y_pred, C = model(image_batch, phi_test)
    class_unseen = Classifier_Unseen(W, C)
    classification = class_unseen(phi, scores)
    #print("Current TestLoss: {}".format(loss))
    return classification


if __name__ == '__main__':
    tf.compat.v1.enable_eager_execution()
    gpu = tf.config.experimental.list_physical_devices('GPU')
    print("Num GPUs Available: ", len(gpu))
    if len(gpu) > 0:
        tf.config.experimental.set_memory_growth(gpu[0], True)
        tf.config.experimental.set_memory_growth(gpu[1], True)

    # read dataset
    path_root = os.path.abspath(os.path.dirname(__file__))
    bird_data = DataSet("/Volumes/Watermelon")# DataSet(path_root)
    phi_train = bird_data.get_phi(set=0)
    w = bird_data.get_w(alpha=1)  # (50*150)
    train_class_list, test_class_list = bird_data.get_class_split(mode="easy")
    train_ds, test_ds = bird_data.load_gpu(batch_size=4)

    #path_root = os.path.abspath(os.path.dirname(__file__))
    #database = DataSet("/Volumes/Watermelon")  # path_root)
    #PHI = database.get_phi()
    #DS, DS_test = database.load_gpu(batch_size=5)  # image_batch, label_batch
    modelaki = FinalModel()

    # define loss and opt functions
    loss_fun = Loss().final_loss
    step = tf.Variable(0, trainable=False)
    boundaries = [187 * 5, 187 * 10]
    values = [0.05, 0.005, 0.0005]
    learning_rate_fn = PiecewiseConstantDecay(boundaries, values)
    # Later, whenever we perform an optimization step, we pass in the step.
    learning_rate = learning_rate_fn(step)
    opt_fun = tfa.optimizers.SGDW(learning_rate=learning_rate, weight_decay=5 * 1e-4, momentum=0.9)
    # opt_fun = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9)

    # define checkpoint settings
    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=opt_fun, net=modelaki)
    manager = tf.train.CheckpointManager(ckpt, path_root + '/tf_ckpts',
                                         max_to_keep=10)  # keep only the three most recent checkpoints
    ckpt.restore(manager.latest_checkpoint)  # pickup training from where you left off

    # define train and test loss and accuracy
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.Accuracy(name='train_accuracy')
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.Accuracy(name='test_accuracy')

    # define data for tensorboard
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
    test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    EPOCHS = 500
    CHECKEPOCHS = 5

    count = 0
    # run for each epoch and batch
    # loss and accuracy are saved every 50 updates
    # model saved every 3 epochs
    for epoch in range(EPOCHS):
        train_loss_results = []
        train_accuracy_results = []
        for images, labels in train_ds:
            if images.shape[0] == BATCH_SIZE:
                train_step(modelaki, images, labels, phi_train, loss_fun, opt_fun, epoch)
                train_loss_results.append(train_loss.result())
                train_accuracy_results.append(train_accuracy.result())
                count += 1
                if count % 50 == 0:
                    template = 'Count {}, Loss: {}, Accuracy: {}'
                    print(template.format(count + 1, train_loss, train_accuracy))
                    with open(path_root + '/log.txt', 'a') as temp:
                        temp.write('Epoch: {}, step: {}, train_Loss: {}, train_Accuracy: {}\n'.format(
                            epoch + 1, count, sum(train_loss_results) / len(train_accuracy_results),
                            sum(train_accuracy_results) / len(train_accuracy_results)))
        ckpt.step.assign_add(1)
        if int(ckpt.step) % CHECKEPOCHS == 0:
            save_path = manager.save()

        template = 'Epoch {}, Loss: {}, Accuracy: {}'
        print(template.format(epoch + 1, train_loss, train_accuracy))

    # TEST UNSEEN CLASSES
    phi_test = bird_data.get_phi(set=0)
    for images, labels in test_ds:
        classification = test_step(modelaki, images, w, phi_test)
