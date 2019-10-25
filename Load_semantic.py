import tensorflow as tf


file_path = '/Volumes/Watermelon/CUB_200_2011/attributes.txt'

attributes = {}
with open(file_path) as fp:
    for cnt, line in enumerate(fp):
        id_a = line.split(" ")[0]
        info = line.split(" ")[1].split("::")
        if info[0] in attributes.keys():
            attributes[info[0]] += [int(id_a)]
        else:
            attributes[info[0]] = [int(id_a)]

file_path2 = '/Volumes/Watermelon/CUB_200_2011/CUB_200_2011/attributes/image_attribute_labels.txt'

import numpy as np

birds_at = {}
n_att = len(attributes.keys())

nameatt_id = {}
for i, key in enumerate(attributes.keys()):
    nameatt_id[key] = i

with open(file_path2) as fp:
    for cnt, line in enumerate(fp):
        id_bird = line.split(" ")[0]
        if id_bird not in birds_at.keys():
            birds_at[id_bird] = np.zeros(n_att)

        id_att = int(line.split(" ")[1])
        present = int(line.split(" ")[2])
        if present:
            for i, key in enumerate(attributes.keys()):
                if id_att in attributes[key]:
                    birds_at[id_bird][i] += np.where(np.array(attributes[key]) == id_att)[0][0]

birds_semantics = []
for key in birds_at.keys():
    birds_semantics.append(tf.convert_to_tensor(birds_at[key]))
birds_semantics = tf.stack(birds_semantics)

print("hola")

