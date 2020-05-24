import glob
import sys
import os
import pickle
import random
import collections
import numpy as np
import json
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn import metrics
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten,MaxPooling2D
from keras.models import load_model, Model
from keras import optimizers
from keras.optimizers import Adam
from keras.utils import np_utils, to_categorical
import matplotlib.pyplot as plt
from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean
from skimage.io import imread, imshow
import pandas as pd
from keras.applications.resnet_v2 import ResNet50V2
from keras.preprocessing import image
import kerasModel
from DataGeneratorClass import My_Custom_Generator
from prettytable import PrettyTable, PLAIN_COLUMNS
import custom_model_eval
from matplotlib import pyplot
import talos
import custom_model
print('checkkkk')
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
print('checkkkk')
random.seed(1)
#=======================================================================================
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'trained_model.h5'
batch_size = 64
input_shape = (40, 40, 3)
#=======================================================================================
#TODO:fix label to integer dict
'''load labels to integer dict and integer to label dict for later use after model.predict'''
with open('label_to_integer.json') as a:
    label_to_integer = json.load(a)

with open('integer_to_label.json') as b:
    integer_to_label = json.load(b)

with open('labels_list.json') as b:
    labels_list = json.load(b)

print('dict of label to integer: ',label_to_integer)
print('dict of integer to label: ',integer_to_label)
print('labels list: ',labels_list)
#=======================================================================================
num_classes = len(labels_list)



'''load train, test, validation'''
X_train_filenames = np.load('X_train_filenames.npy')
y_train = np.load('y_train.npy')
X_test_filenames = np.load('X_test_filenames.npy')
y_test = np.load('y_test.npy')
X_val_filenames = np.load('X_val_filenames.npy')
y_val = np.load('y_val.npy')

print('X_train_filenames.shape: ', X_train_filenames.shape)  # (3800,) example
print('y_train.shape: ', y_train.shape)  # (3800, 12) example

print('X_test_filenames.shape: ', X_test_filenames.shape)  # (950,) example
print('y_test.shape: ', y_test.shape)  # (950, 12) example

print('X_val_filenames.shape: ', X_val_filenames.shape)  # (950,) example
print('y_val.shape: ', y_val.shape)

print('X_train_filenames: ',X_test_filenames[:5])
print('y_test: ',y_test[:5])


'''send train and test to the Generator'''
my_training_batch_generator = My_Custom_Generator(X_train_filenames, y_train, batch_size,input_shape)
my_test_batch_generator = My_Custom_Generator(X_test_filenames, y_test, batch_size, input_shape)
my_val_batch_generator = My_Custom_Generator(X_val_filenames, y_val, batch_size, input_shape)

y_train_check = np.argmax(y_train, axis=1)
y_test_check = np.argmax(y_test, axis=1)
y_val_check = np.argmax(y_val, axis=1)
print('X_train_filenames first 20: ',X_train_filenames[:20])
print('y_train after argmax first 100 :', y_train_check[:100])
print('X_test_filenames first 20: ',X_test_filenames[:20])
print('y_test after argmax first 100 :', y_test_check[:100])
print('X_val_filenames first 20: ',X_val_filenames[:20])
print('y_val after argmax first 100 :', y_val_check[:100])

'''build model or if model exists lets predict and run evaluations'''
# if glob.glob(save_dir+'/*'):#TODO: make this work with the new architecture
build = False
if build == False:
    # model_name = 'trained_model_resnext101.h5'
    # my_model = load_model(save_dir+'/'+model_name)
    # n_test = my_test_batch_generator.getNumber()
    # y_pred1 = Model.predict_generator(my_model,my_test_batch_generator,n_test/batch_size)
    #
    # score, acc = Model.evaluate_generator(my_model,my_test_batch_generator,
    #                             batch_size)
    # print('Test score:', score)
    # print('Test accuracy:', acc)
    # print('Score: ',score)
    # print('y_pred1 before argmax :', y_pred1[:100])
    # print('y_pred1 before len: ', len(y_pred1))
    # print('y_test before argmax :', y_test)
    # print('y_test before len: ', len(y_test))
    # y_pred = np.argmax(y_pred1, axis=1)
    # y_test = np.argmax(y_test, axis=1)
    # print('y_pred1 after argmax first 100 :', y_pred[:100])
    # print('y_pred1 after len: ', len(y_pred))
    # print('y_test after argmax first 100 :', y_test[:100])
    # print('y_test after len: ', len(y_test))
    #
    # print('f1_score, macro: ', metrics.f1_score(y_test, y_pred, average="macro"))
    # print('f1_score, micro: ', metrics.f1_score(y_test, y_pred, average="micro"))
    #
    # print('confusion matrix: ', metrics.confusion_matrix(y_test, y_pred))
    #
    # img = resize(imread('akoya1.png'), input_shape)
    # img = np.asarray(img)
    # plt.imshow(img)
    # img = np.expand_dims(img, axis=0)
    #
    # img2 = resize(imread('freshwater.png'), input_shape)
    # img2 = np.asarray(img2)
    # plt.imshow(img2)
    # plt.show()
    # img2 = np.expand_dims(img2, axis=0)
    #
    # print('prediction on 1 akoya img.....')
    # output = my_model.predict(img)
    # print('real label to int: ', label_to_integer['akoya'])
    # print('predicted integer label before argmax:', output)
    # output = np.argmax(output)
    # print('predicted integer label after argmax: ', output)
    # print('int to label conversion: ', integer_to_label[str(output)])
    #
    # print('prediction on 1 freshwater img.....')
    # output = my_model.predict(img2)
    # print('real label to int: ', label_to_integer['freshwater'])
    # print('predicted integer label before argmax:', output)
    # output = np.argmax(output)
    # print('predicted integer label after argmax: ', output)
    # print('int to label conversion: ', integer_to_label[str(output)])
    #
    # print(y_test[:5])
    # print(y_pred[:5])
    with open(save_dir+'/'+'trained_model_resnext101_6_epoch20.h5_history','rb') as b:
        history = pickle.load(b)
    print(history.history.keys())

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    ax1.plot(history.history['loss'], color='b', label="Training loss")
    ax1.plot(history.history['val_loss'], color='r', label="validation loss")
    ax1.set_xticks(np.arange(1, 5, 1))
    ax1.set_yticks(np.arange(0, 1, 0.1))

    ax2.plot(history.history['accuracy'], color='b', label="Training accuracy")
    ax2.plot(history.history['val_accuracy'], color='r', label="Validation accuracy")
    ax2.set_xticks(np.arange(1, 5, 1))

    legend = plt.legend(loc='best', shadow=True)
    plt.tight_layout()
    plt.show()




else:
    # print("Buidling VGG16 model......")
    # kerasModel.pearl_type_model_vgg16(my_training_batch_generator
    #                             ,my_val_batch_generator
    #                             ,save_dir,batch_size,input_shape)

    # print("Buidling ResNet50V2 model......")
    # kerasModel.pearl_type_model_resnet50v2(my_training_batch_generator
    #                                   , my_val_batch_generator
    #                                   , save_dir,batch_size, input_shape)

    print("Buidling ResNext101 model......")
    kerasModel.pearl_type_model_resnext101(my_training_batch_generator
                                           , my_val_batch_generator
                                           , save_dir, batch_size, input_shape)


    # print('Building Custom NN eval....')
    # custom_model_eval.run_custom_model(my_training_batch_generator,my_val_batch_generator,input_shape)
    # talos_parameter_eval = custom_model_eval.load_object('example.pickle')
    # custom_model_eval.print_hyperparameter_search_stats(talos_parameter_eval)
    # custom_model_eval.print_eval_table(talos_parameter_eval)
