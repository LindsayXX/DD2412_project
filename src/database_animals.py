import tensorflow as tf
import os
from PIL import Image
import numpy as np

'''
im = Image.open("/Users/stella/Downloads/Animals_with_Attributes2_3/JPEGImages/hippopotamus/hippopotamus_10003.jpg")
width, height = im.size #((1024, 684))
im=im.resize((128,128),resample=0)
#im.show()
#display.display(Image.open(str('/Users/stella/Downloads/Animals_with_Attributes2_3/JPEGImages/hippopotamus/hippopotamus_10003.jpg')))
'''

# Reading the dataset
def image_paths(dataset_path):
    """
    input: dpath of the whole image dataset
    returns: all of the imagepaths and all of the labels/classes
    as type tensorflow.python.framework.ops.EagerTensor
    """

    imagepaths = []
    # os.walk returns a tuple of three elements: (root_dir_path, sub_dirs, files)

    # picking up the names of the subfiles equals the names of the classes
    classes = os.walk(dataset_path).__next__()[1]

    # List each sub-directory (the classes)
    for c in classes:
        # seperate animal folder(=class)
        c_dir = os.path.join(dataset_path, c)
        # make a walk object for each folder
        walk = os.walk(c_dir).__next__()
        # Add each image (walk[2]) to the imagepaths list
        for sample in walk[2]:
            imagepath = os.path.join(c_dir, sample)
            imagepaths.append(imagepath)

    # cpnvert to tensfor
    # imagepaths = tf.convert_to_tensor(imagepaths, dtype=tf.string)

    return imagepaths


#imagepaths= image_paths('/Users/stella/Downloads/Animals_with_Attributes2_3/JPEGImages')

#get the label of one image

def get_label(file_path):
  # convert the path to a list of path components
  parts = tf.strings.split(file_path, '/')
  # The second to last is the class-directory
  return parts[-2]

#get_label('/Users/stella/Downloads/Animals_with_Attributes2_3/JPEGImages/hippopotamus/hippopotamus_10003.jpg')

#decode one image

[IMG_WIDTH, IMG_HEIGHT]=[128,128]
def decode_img(img):
  # convert the compressed string to a 3D uint8 tensor
  img = tf.image.decode_jpeg(img, channels=3)
  # Use `convert_image_dtype` to convert to floats in the [0,1] range.
  img = tf.image.convert_image_dtype(img, tf.float32)
  # resize the image to the desired size.
  return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])

#return img,label for one image

def process_path(file_path):
  label = get_label(file_path)
  # load the raw data from the file as a string
  img = tf.io.read_file(file_path)
  img = decode_img(img)
  return img, label

#process_path('/Users/stella/Downloads/Animals_with_Attributes2_3/JPEGImages/hippopotamus/hippopotamus_10003.jpg')

# Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
list_ds = tf.data.Dataset.list_files(str('/Users/stella/Downloads/Animals_with_Attributes2_3/JPEGImages/*/*'))
labeled_ds = list_ds.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)

for image, label in labeled_ds.take(1):
  print("Image shape: ", image.numpy().shape)
  print("Label: ", label.numpy())

