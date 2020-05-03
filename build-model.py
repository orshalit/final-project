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
random.seed(0)
#=======================================================================================
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'trained_model.h5'
labels = images.getLabels()
batch_size = 64
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


'''send train and test to the Generator'''
my_training_batch_generator = My_Custom_Generator(X_train_filenames, y_train, batch_size,input_shape)
my_test_batch_generator = My_Custom_Generator(X_test_filenames, y_test, batch_size, input_shape)
my_val_batch_generator = My_Custom_Generator(X_val_filenames, y_val, batch_size, input_shape)


'''build model or if model exists lets predict and run evaluations'''
if glob.glob(save_dir+'/*'):#TODO: make this work with the new architecture
    # my_model = load_model(save_dir+'/'+model_name)
    my_model = load_model('img_aug_cnn_restnet50.h5')
    n_test = my_test_batch_generator.getNumber()
    y_pred1 = Model.predict_generator(my_model,my_test_batch_generator,n_test/batch_size)
    # score = Model.evaluate_generator(my_model, my_test_batch_generator, n_test // batch_size)
    # print('Score: ',score)

    print('y_pred1 :', y_pred1)
    y_pred = np.argmax(y_pred1, axis=1)
    y_test = np.argmax(y_test, axis=1)
    # print('y_pred :', y_pred)
    # print('y_test :', y_test)
    # print('y_pred1 len: ', len(y_pred1))
    # print('y_pred len: ',len(y_pred))
    # print('y_test len: ',len(y_test))
    # Print f1, precision, and recall scores
    # print(metrics.precision_score(y_test, y_pred, average="micro"))
    # print(metrics.recall_score(y_test, y_pred, average="micro"))
    print(metrics.f1_score(y_test, y_pred, average="macro"))
    print(metrics.f1_score(y_test, y_pred, average="micro"))
    #
    print(metrics.confusion_matrix(y_test, y_pred))

    # img = image.load_img("akoya1.png", target_size=(40, 40))
    # img = np.asarray(img)
    # plt.imshow(img)
    # img = np.expand_dims(img, axis=0)
    #
    #
    # output = my_model.predict(img)
    # output = np.argmax(output)
    # print(output)
    # print(integer_to_label[str(output)])
    #
    # restnet = ResNet50V2(include_top=False, weights=None, input_shape=input_shape,classes=len(integer_to_label))
    # output = restnet.layers[-1].output
    # output = keras.layers.Flatten()(output)
    # restnet = Model(restnet.input, output=output)
    # for layer in restnet.layers:
    #     layer.trainable = False
    # restnet.summary()
    # model = Sequential()
    # model.add(restnet)
    # model.add(Dense(512, activation='relu', input_dim=input_shape))
    # model.add(Dropout(0.3))
    # model.add(Dense(512, activation='relu'))
    # model.add(Dropout(0.3))
    # model.add(Dense(4, activation='softmax'))
    # model.compile(loss=keras.losses.categorical_crossentropy,
    #               optimizer=optimizers.RMSprop(lr=2e-5),
    #               metrics=['accuracy'])
    # model.summary()
    #
    # history = model.fit_generator(my_training_batch_generator,
    #                               steps_per_epoch=100,
    #                               epochs=2,
    #                               validation_data=my_val_batch_generator,
    #                               validation_steps=50,
    #                               verbose=1)
    # model.save('img_aug_cnn_restnet50.h5')

else:
    print("Buidling VGG16 model......")
    kerasModel.pearl_type_model_vgg16(my_training_batch_generator
                                ,my_val_batch_generator
                                ,save_dir,model_name,batch_size,input_shape)

    print("Buidling ResNet50V2 model......")
    kerasModel.pearl_type_model_vgg16(my_training_batch_generator
                                      , my_val_batch_generator
                                      , save_dir, model_name, batch_size, input_shape)


