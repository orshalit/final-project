import keras
from keras.preprocessing import image
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Flatten, Dense
from keras.activations import relu, elu
from DataGeneratorClass import My_Custom_Generator


from talos.model import network_shape
from talos.model import hidden_layers

class __Config__(object):
    pass
config =__Config__()
config.optimizer =Adam
config.optimizer_parameters ={'lr' :0.0001 ,'decay' :0.001}
config.loss ='categorical_crossentropy'
config.metric =['accuracy']


def create_model(training_set, validation_set, verbose=False):
    def _create_conv_shape_(params):
        def shape(params):
            if params['hidden_layers'] == 1:
                return [params['first_neuron']]
            if params['hidden_layers'] == 2:
                return [params['first_neuron'], params['last_neuron']]
            else:
                params = params.copy()
                params['hidden_layers'] -= 2
                s_list = network_shape.network_shape(params, params['last_neuron'])
                return [params['first_neuron'], *s_list, params['last_neuron']]

        conv_depth_params = {
            'hidden_layers': params['conv_hidden_layers'],
            'shapes': params['conv_depth_shape'],
            'first_neuron': params['conv_depth_first_neuron'],
            'last_neuron': params['conv_depth_last_neuron'],
        }
        conv_size_params = {
            'hidden_layers': params['conv_hidden_layers'],
            'shapes': params['conv_size_shape'],
            'first_neuron': params['conv_size_first_neuron'],
            'last_neuron': params['conv_size_last_neuron'],
        }

        conv_depth_shape = shape(conv_depth_params)
        conv_size_shape = shape(conv_size_params)
        conv_shape = zip(conv_depth_shape, conv_size_shape)

        return conv_shape

    def model(dummyXtrain, dummyYtrain, dummyXval, dummyYval, params):
        conv_shape = _create_conv_shape_(params)

        model = Sequential()

        for i, (depth, size) in enumerate(conv_shape):
            if i == 0:
                model.add(Conv2D(depth, size, input_shape=params['input_shape']))
            else:
                model.add(Conv2D(depth, size))
            model.add(Activation('relu'))

        model.add(Flatten())

        hidden_layers(model, params, params['last_neuron'])

        model.add(Dense(4))     #4 is the shape of the data
        model.add(Activation('softmax'))

        global config
        optimizer = config.optimizer(**config.optimizer_parameters)
        model.compile(loss=config.loss,
                      optimizer=optimizer,
                      metrics=config.metric)

        training_set.batch_size = params['batch_size']
        validation_set.batch_size = params['batch_size']

        model.summary()

        n_train = My_Custom_Generator.getNumber(training_set)
        n_test = My_Custom_Generator.getNumber(validation_set)
        print('number of training images: ', n_train)
        print('number of val images: ', n_test)



        history = model.fit_generator(
            training_set,
            validation_data=validation_set,
            epochs=params['epoch'],
            verbose=int(params['verbose']),
        )
        #TODO: save model as h5 and history using round as naming index? maybe...
        return history, model
        #	model.permutation_filter=lambda params: permutation_filter(training_set.shape,params)

    return model

