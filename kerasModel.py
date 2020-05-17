import numpy as np
import pandas as pd

import os
import pickle
from skimage.io import imread
from skimage.transform import resize

import keras
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Conv1D, Conv2D, MaxPooling1D, MaxPool2D, Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.utils import to_categorical
from keras.applications import xception
from keras.applications.resnet_v2 import ResNet50V2
from keras_applications.resnet_common import ResNeXt101

from DataGeneratorClass import My_Custom_Generator
from keras import optimizers

#tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
earlystop = EarlyStopping(patience=10)
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
                                            patience=2,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.00001)
callbacks = [earlystop, learning_rate_reduction]

def pearl_type_model_vgg16(my_training_batch_generator, my_test_batch_generator, save_dir=os.path.join(os.getcwd(), 'saved_models')
                     , batch_size=32
                     , input_shape=(40,40,3)):
    model_name = 'vgg16.h5'
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
    history = model.fit_generator(generator=my_training_batch_generator,
                        steps_per_epoch = int(n_train // batch_size),
                        epochs = 1,
                        verbose = 1,
                        validation_data = my_test_batch_generator,
                        validation_steps = int(n_test // batch_size))

    hist_df = pd.DataFrame(history.history)


    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, model_name)
    print('saves this model?!')
    model.save(model_path)
    print('Saved trained model at %s ' % model_path)

    # save to pickle:
    hist_file = 'vgg16_history'
    with open(model_path + hist_file, 'wb') as file_pi:
        pickle.dump(history, file_pi)

def pearl_type_model_resnet50v2(my_training_batch_generator, my_test_batch_generator,save_dir=os.path.join(os.getcwd(), 'saved_models')
                         , batch_size=32
                         , input_shape=(40, 40, 3)):
    model_name = 'trained_model_resnet50v2.h5'
    restnet = ResNet50V2(include_top=False, weights=None, input_shape=input_shape, classes=4)
    output = restnet.layers[-1].output
    output = keras.layers.Flatten()(output)
    restnet = Model(restnet.input, output=output)
    for layer in restnet.layers:
        layer.trainable = False
    restnet.summary()
    model = Sequential()
    model.add(restnet)
    model.add(Dense(512, activation='relu', input_dim=input_shape))
    # model.add(Dropout(0.3))
    model.add(Dense(512, activation='relu'))
    # model.add(Dropout(0.3))
    model.add(Dense(4, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=optimizers.adam(lr=2e-5),
                  metrics=['accuracy'])
    model.summary()

    n_train = My_Custom_Generator.getNumber(my_training_batch_generator)
    n_test = My_Custom_Generator.getNumber(my_test_batch_generator)
    print('number of training images: ', n_train)
    print('number of val images: ', n_test)

    history = model.fit_generator(my_training_batch_generator,
                                  steps_per_epoch=1,#int(n_train // batch_size),
                                  epochs=1,
                                  validation_data=my_test_batch_generator,
                                  validation_steps=1,#int(n_test // batch_size),
                                  #callbacks=[tbCallBack],
                                  verbose=1)

    hist_df = pd.DataFrame(history.history)

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    print('type of model name: ',model_name)
    model_path = os.path.join(save_dir, str(model_name))
    model.save(model_path)
    print('Saved trained model at %s ' % model_path)

    # save to pickle:
    hist_file = 'resnext50v2_history'
    with open(model_path + hist_file, 'wb') as file_pi:
        pickle.dump(history, file_pi)

def pearl_type_model_resnext101(my_training_batch_generator, my_test_batch_generator,save_dir=os.path.join(os.getcwd(), 'saved_models')
                         , batch_size=32
                         , input_shape=(40, 40, 3)):
    model_name = 'trained_model_resnext101.h5'
    restnet = ResNeXt101(include_top=False, weights=None, input_shape=input_shape, classes=4
                         ,backend = keras.backend
                         , layers = keras.layers
                         , models = keras.models
                         , utils = keras.utils)
    output = restnet.layers[-1].output
    output = keras.layers.Flatten()(output)
    restnet = Model(restnet.input, output=output)
    #for layer in restnet.layers:
    #    layer.trainable = False
    restnet.summary()
    model = Sequential()
    model.add(restnet)
    model.add(Dense(512, activation='relu', input_dim=input_shape))

    # model.add(Dropout(0.3))
    model.add(Dense(512, activation='relu'))
    # model.add(Dropout(0.3))
    model.add(Dense(4, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.adam(lr=0.01),
                  metrics=['accuracy'])
    model.summary()

    n_train = My_Custom_Generator.getNumber(my_training_batch_generator)
    n_test = My_Custom_Generator.getNumber(my_test_batch_generator)
    print('number of training images: ', n_train)
    print('number of val images: ', n_test)

    history = model.fit_generator(my_training_batch_generator,
                                  steps_per_epoch=1,  # int(n_train // batch_size),
                                  epochs=1,
                                  validation_data=my_test_batch_generator,
                                  validation_steps=1,  # int(n_test // batch_size),
                                  #callbacks=[tbCallBack],
                                  verbose=1)

    #hist_df = pd.DataFrame(history.history)

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    print('type of model name: ', model_name)
    model_path = os.path.join(save_dir, str(model_name))
    model.save(model_path)
    print('Saved trained model at %s ' % model_path)

    hist_file = 'resnext101_history'
    with open(model_path + hist_file, 'wb') as file_pi:
        pickle.dump(history, file_pi)
    # # save to json:
    # hist_json_file = 'resnext101_history.json'
    # with open(model_path + hist_json_file, mode='w') as f:
    #     hist_df.to_json(f)
