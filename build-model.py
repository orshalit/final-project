import glob
import os
import imageAugmentor as images
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
from keras.optimizers import Adam
from keras.utils import np_utils, to_categorical
import matplotlib.pyplot as plt
from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean
from skimage.io import imread, imshow
import pandas as pd
from keras.preprocessing import image
import kerasModel
from DataGeneratorClass import My_Custom_Generator
random.seed(0)
#=======================================================================================
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'trained_model.h5'
labels = images.getLabels()
batch_size = 32
input_shape = (40, 40, 3)
num_classes = len(labels)
#=======================================================================================

'''load labels to integer dict and integer to label dict for later use after model.predict'''
with open('label_to_integer.json') as a:
    label_to_integer = json.load(a)

with open('integer_to_label.json') as b:
    integer_to_label = json.load(b)

print('dict of label to integer: ',label_to_integer)
print('dict of integer to label: ',integer_to_label)

#=======================================================================================




'''load train, test, validation'''
X_train_filenames = np.load('X_train_filenames.npy')
y_train = np.load('y_train.npy')
X_test_filenames = np.load('X_test_filenames.npy')
y_test = np.load('y_test.npy')
X_val_filenames = np.load('X_val_filenames.npy')
y_val = np.load('y_val.npy')

print(len(y_val))
exit(1)
'''send train and test to the Generator'''
my_training_batch_generator = My_Custom_Generator(X_train_filenames, y_train, batch_size,input_shape)
my_test_batch_generator = My_Custom_Generator(X_test_filenames, y_test, batch_size, input_shape)
my_val_batch_generator = My_Custom_Generator(X_val_filenames, y_val, batch_size, input_shape)


'''build model or if model exists lets predict and run evaluations'''
if glob.glob(save_dir+'/*'):#TODO: make this work with the new architecture
    my_model = load_model(save_dir+'/'+model_name)
    # n_test = my_test_batch_generator.getNumber()
    # y_pred1 = Model.predict_generator(my_model,my_test_batch_generator,n_test//batch_size)
    # print('y_pred1 :', y_pred1)
    # y_pred = np.argmax(y_pred1, axis=1)
    # print('y_pred :', y_pred)
    # # Print f1, precision, and recall scores
    # print(metrics.precision_score(y_test, y_pred, average="micro"))
    # print(metrics.recall_score(y_test, y_pred, average="micro"))
    # print(metrics.f1_score(y_test, y_pred, average="weighted"))
    #
    # print(metrics.confusion_matrix(y_test, y_pred))

    img = image.load_img("akoya1.png", target_size=(224, 224))
    img = np.asarray(img)
    plt.imshow(img)
    img = np.expand_dims(img, axis=0)


    output = my_model.predict(img)
    output = np.argmax(output)
    print(output)
    print(integer_to_label[str(output)])


else:
    print("Buidling model......")
    kerasModel.pearl_type_model(my_training_batch_generator
                                ,my_val_batch_generator
                                ,save_dir,model_name,batch_size,input_shape)


