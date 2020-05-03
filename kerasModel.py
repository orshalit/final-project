import numpy as np
import pandas as pd

import os

from skimage.io import imread
from skimage.transform import resize

import keras
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Conv1D, Conv2D, MaxPooling1D, MaxPool2D, Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.utils import to_categorical
from keras.applications.resnet_v2 import ResNet50V2
from DataGeneratorClass import My_Custom_Generator
from keras import optimizers

def pearl_type_model_vgg16(my_training_batch_generator, my_test_batch_generator, save_dir=os.path.join(os.getcwd(), 'saved_models')
                     , model_name='trained_model.h5'
                     , batch_size=32
                     , input_shape=(40,40,3)):


    model = Sequential()
    model.add(Conv2D(input_shape=input_shape,filters=64,kernel_size=(3,3),padding="same", activation="relu"))
    model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Flatten())
    model.add(Dense(units=4096,activation="relu"))
    model.add(Dense(units=4096,activation="relu"))
    model.add(Dense(units=4, activation="softmax"))
    opt = Adam(lr=0.01 )
    model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

    model.summary()


    n_train = My_Custom_Generator.getNumber(my_training_batch_generator)
    n_test = My_Custom_Generator.getNumber(my_test_batch_generator)
    print('number of training images: ',n_train)
    print('number of val images: ', n_test)
    model.fit_generator(generator=my_training_batch_generator,
                        steps_per_epoch = int(n_train // batch_size),
                        epochs = 10,
                        verbose = 1,
                        validation_data = my_test_batch_generator,
                        validation_steps = int(n_test // batch_size))

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, model_name)
    model.save(model_path)
    print('Saved trained model at %s ' % model_path)

def pearl_type_model_resnet50v2(my_training_batch_generator, my_test_batch_generator,save_dir=os.path.join(os.getcwd(), 'saved_models')
                         , model_name='trained_model_resnet50v2.h5'
                         , batch_size=32
                         , input_shape=(40, 40, 3)):
    restnet = ResNet50V2(include_top=False, weights=None, input_shape=input_shape, classes=len(integer_to_label))
    output = restnet.layers[-1].output
    output = keras.layers.Flatten()(output)
    restnet = Model(restnet.input, output=output)
    for layer in restnet.layers:
        layer.trainable = False
    restnet.summary()
    model = Sequential()
    model.add(restnet)
    model.add(Dense(512, activation='relu', input_dim=input_shape))
    model.add(Dropout(0.3))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(4, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=optimizers.RMSprop(lr=2e-5),
                  metrics=['accuracy'])
    model.summary()

    history = model.fit_generator(my_training_batch_generator,
                                  steps_per_epoch=100,
                                  epochs=2,
                                  validation_data=my_test_batch_generator,
                                  validation_steps=50,
                                  verbose=1)
    model.save('trained_model_resnet50v2.h5')

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, model_name)
    model.save(model_path)
    print('Saved trained model at %s ' % model_path)
