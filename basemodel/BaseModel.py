from dataloader import DataSet
import os
import tensorflow as tf
import tensorflow_addons as tfa

IMG_SIZE = 448
BATCH_SIZE = 32

tf.random.set_seed(3)

class BaseModel(tf.keras.Model):
    '''
    Baseline Model: without multi-attention subnet and class-center triplet loss
    '''

    def __init__(self, semantic_size):
        super(BaseModel, self).__init__()
        #self.vgg_features = tf.keras.applications.VGG19(input_shape=(IMG_SIZE/2, IMG_SIZE/2, 3), include_top=False,
        #                                                weights='imagenet')
        #self.gp = tf.keras.layers.GlobalAveragePooling2D()
        self.vgg_features_new = tf.keras.applications.VGG19(input_shape=(IMG_SIZE / 2, IMG_SIZE / 2, 3), include_top=False,
                                                        pooling='avg', weights='imagenet')
        self.vgg_features_new.trainable = False
        #w_init = tf.random_normal_initializer(0, 0.01)  # default: (0,0.05)
        #self.W = tf.Variable(initial_value=w_init(shape=(512, semantic_size), dtype='float32'), trainable=True,
        #                     name="W")
        self.W = tf.keras.layers.Dense(semantic_size, name="W")
        # c_init = tf.random_normal_initializer(0, 0.01)
        # self.C = tf.Variable(initial_value=c_init(shape=(semantic_size, n_class), dtype='float32'), trainable=True,name="C")

    @tf.function
    def call(self, x, phi):
        resized = tf.image.resize(x, (224, 224))  # δίνει (32,224,224,3)
        #features = self.vgg_features(resized)  # δίνει (32,7,7,512)
        #global_theta = self.gp(features)  # tensor of shape (32,512)
        global_theta = self.vgg_features_new(resized)
        #global_mapped = tf.linalg.matmul(global_theta, self.W)
        global_mapped = self.W(global_theta)
        global_scores = tf.linalg.matmul(global_mapped, phi)
        # out = self.fc(out)  #prediction (batch_size, 200)

        #return tf.math.l2_normalize(global_scores)  # , out
        return global_scores

@tf.function
def loss_baseline(score, labels):
    # scores:  batch_size*200*1
    # labels: batch_size * 1
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=score))



if __name__ == '__main__':
    #os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    #import tensorflow as tf
    tf.compat.v1.enable_eager_execution()
    gpu = tf.config.experimental.list_physical_devices('GPU')
    print("Num GPUs Available: ", len(gpu))
    if gpu:
        tf.config.experimental.set_memory_growth(gpu[0], True)
    path_root = os.path.abspath(os.path.dirname(__file__))  # '/content/gdrive/My Drive/data'
    bird_data = DataSet(path_root)
    # load all imgs
    phi = bird_data.get_phi()
    train_ds, test_ds = bird_data.load_gpu(batch_size=BATCH_SIZE)
    # only take 1000 images for local test
    #train_ds = bird_data.load(GPU=False, train=True, batch_size=32)
    #test_ds = bird_data.load(GPU=False, train=False, batch_size=32)

    #image_batch, label_batch = next(iter(train_ds))

    model = BaseModel(150, 312)
    #opt = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9)  # or SGDW with weight decay
    opt = tfa.optimizers.SGDW(
        learning_rate=0.0001, weight_decay=5 * 1e-4, momentum=0.9)

    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=opt, net=model)
    manager = tf.train.CheckpointManager(ckpt, path_root + '/tf_ckpts',
                                         max_to_keep=3)  # keep only the three most recent checkpoints
    ckpt.restore(manager.latest_checkpoint)  # pickup training from where you left off

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.Accuracy(name='train_accuracy')
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.Accuracy(name='test_accuracy')

    @tf.function
    def test_step(images, labels):
        scores = model(images, phi)
        t_loss = loss_baseline(scores, labels)
        t_pred = tf.math.argmax(scores, axis=1)

        test_loss(t_loss)
        test_accuracy(tf.expand_dims(labels, -1), tf.expand_dims(t_pred, -1))

    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            scores = model(images, phi)
            loss = loss_baseline(scores, labels)
        gradients = tape.gradient(loss, model.trainable_variables)
        # print(gradients)
        opt.apply_gradients(zip(gradients, model.trainable_variables))
        y_pred = tf.math.argmax(scores, axis=1)

        train_loss(loss)
        train_accuracy(tf.expand_dims(labels, -1), tf.expand_dims(y_pred, -1))

    EPOCHS = 50
    CHECKEPOCHS = 1
    print("Here it begins")
    for epoch in range(EPOCHS):
        train_loss_results = []
        train_accuracy_results = []
        test_loss_results = []
        test_accuracy_results = []
        count = 0
        for images, labels in train_ds:
        # check none value
        #images, labels = next(iter(train_ds))
            train_step(images, labels)
            count += 1
            train_loss_results.append(train_loss.result())
            train_accuracy_results.append(train_accuracy.result())
            if count % 50 == 0:
                tf.print('Epoch {}, train_Loss: {}, train_Accuracy: {}\n'.format(epoch + 1, train_loss.result(), train_accuracy.result()))

        #for test_images, test_labels in test_ds:
            #test_images, test_labels = next(iter(test_ds))
            #test_step(test_images, test_labels)
            #test_loss_results.append(test_loss.result())
            #test_accuracy_results.append(test_accuracy.result())

        ckpt.step.assign_add(1)
        if int(ckpt.step) % CHECKEPOCHS == 0:
            save_path = manager.save()
            with open(path_root + '/log.txt', 'a') as temp:
                temp.write('Epoch {}, train_Loss: {}, train_Accuracy: {}\n'.format(
                    epoch + 1, sum(train_loss_results)/len(train_accuracy_results), sum(train_accuracy_results)/len(train_accuracy_results)))
                #, test_loss.result(), test_accuracy.result())) 
    """
    for test_images, test_labels in test_ds:
        test_step(test_images, test_labels)
    tf.print('Test_Loss: {}, Test_Accuracy: {}\n'.format(test_loss.result(), test_accuracy.result()))
    with open(path_root + '/log.txt', 'a') as temp:
        temp.write('Test_Loss: {}, Test_Accuracy: {}\n'.format(test_loss.result(), test_accuracy.result()))
    """
