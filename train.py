from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
from DataBase import Database
from model import FinalModel
from .LossesLosses import Loss


# IMPORTING DATA
data = Database()
image_batch, label_batch = data()


# MODEL INSTATIATION
model = BaseModel()


#Choose an optimizer and loss function for training:
loss_object = Loss.loss_CCT()
optimizer = tf.keras.optimizers.Adam()



#Select metrics to measure the loss and the accuracy of the model. These metrics accumulate the values over epochs
# and then print the overall result.
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

#Use tf.GradientTape to train the model:
@tf.function
def train_step(image_batch, label_batch):
  with tf.GradientTape() as tape:
    scores = model(image_batch)
    loss = loss_object(scores, label_batch)
  gradients = tape.gradient(loss, model.trainable_variables) #ti einai trainable variables?
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  train_loss(loss)
  train_accuracy(labels, predictions)

#test the model
@tf.function
def test_step(images, labels):
  predictions = model(images)
  t_loss = loss_object(labels, predictions)

  test_loss(t_loss)
  test_accuracy(labels, predictions)


#training!

EPOCHS = 5

for epoch in range(EPOCHS):
  for images, labels in train_ds:
    train_step(images, labels)

  for test_images, test_labels in test_ds:
    test_step(test_images, test_labels)

  template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
  print(template.format(epoch+1,
                        train_loss.result(),
                        train_accuracy.result()*100,
                        test_loss.result(),
                        test_accuracy.result()*100))

  # Reset the metrics for the next epoch
  train_loss.reset_states()
  train_accuracy.reset_states()
  test_loss.reset_states()
  test_accuracy.reset_states()















