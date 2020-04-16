import os
import shutil
import numpy as np
from keras.utils import to_categorical
from sklearn.utils import shuffle
import json
import imageAugmentor as images
from sklearn.model_selection import train_test_split
train_dir = "C:/Users/אור/Desktop/Pearls-Images/train"
dest_dir = "C:/Users/אור/Desktop/Pearls-Images/all_images"
counter = 0

#TODO: make this part generic, move all sub dir images to parent class dir and then delete sub dirs
# for subdir, dirs, files in os.walk(train_dir):
#     # print(files)
#     for file in files:
#         full_path = os.path.join(subdir, file)
#         os.makedirs(os.path.dirname(full_path), exist_ok=True)
#         # print(full_path)
#         # print(dest_dir)
#         shutil.copy2(full_path, dest_dir)
#         counter = counter + 1
# print(counter)
#
#
# #


#TODO:maybe make a function out of this?
# subdirs, dirs, files = os.walk(dest_dir).__next__()
# m = len(files)
# print(m)
#
# filenames = []
# labels = np.zeros((m, 1))
#
#
#
# images_dir = dest_dir
# filenames_counter = 0
# labels_counter = -1
#
# for subdir, dirs, files in os.walk(train_dir):
#     # print(files)
#     for file in files:
#         filenames.append(file)
#         labels[filenames_counter, 0] = labels_counter
#         filenames_counter = filenames_counter + 1
#     labels_counter = labels_counter + 1
#
# print(len(filenames))
# print(labels.shape)
#
#
#
# # saving the filename array as .npy file
# np.save('filenames.npy', filenames)
#
#
#
#
# # One hot vector representation of labels
# y_labels_one_hot = to_categorical(labels)
#
# # saving the y_labels_one_hot array as a .npy file
# np.save('labels.npy', y_labels_one_hot)
#
# filenames = np.load('filenames.npy')
# labels = np.load('labels.npy')
#
# filenames_shuffled, labels_shuffled = shuffle(filenames, labels)
#
# # saving the shuffled file.
# # you can load them later using np.load().
# np.save('labels_shuffled.npy', labels_shuffled)
# np.save('filenames_shuffled.npy', filenames_shuffled)

# labels = images.getLabels()
# label_to_integer = {}
# integer_to_label = {}
# for index, value in enumerate(labels):    #makes class dict for every folder
#     label_to_integer[value] = index
#     integer_to_label[index] = value
#     index = index + 1
#
#


#
# with open('label_to_integer.json', 'w') as fp:
#     json.dump(label_to_integer, fp)
#
# with open('integer_to_label.json', 'w') as fp:
#     json.dump(integer_to_label, fp)



#this is how you load
# with open('label_to_integer.json') as a:
#     a = json.load(a)
#
# with open('integer_to_label.json') as b:
#     b = json.load(b)
# filenames_shuffled = np.load('filenames_shuffled.npy')
# labels_shuffled = np.load('labels_shuffled.npy')


# print(filenames[100])
# print(labels[100])
# print(np.argmax(labels[100]))
# print(len(filenames))
# print(len(labels))

# filenames_shuffled_numpy = np.array(filenames_shuffled)
#
# X_train_filenames, X_val_filenames, y_train, y_val = train_test_split(
#     filenames_shuffled_numpy, labels_shuffled, test_size=0.2, random_state=1)
#
# print(X_train_filenames.shape) # (3800,) example
# print(y_train.shape)           # (3800, 12) example
#
# print(X_val_filenames.shape)   # (950,) example
# print(y_val.shape)             # (950, 12) example
#
# # You can save these files as well. As you will be using them later for training and validation of your model.
# np.save('X_train_filenames.npy', X_train_filenames)
# np.save('y_train.npy', y_train)
#
# np.save('X_test_filenames.npy', X_val_filenames)
# np.save('y_test.npy', y_val)