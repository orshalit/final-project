import glob
import sys
import os
import pickle
import pandas as pd
from DataGeneratorClass import My_Custom_Generator
from prettytable import PrettyTable, PLAIN_COLUMNS

import talos
import custom_model


def set_params(input_shape):
    params = {
            'epoch': [1,1],
            'batch_size': [32],
            'activation': ['relu'],
            # convultion layers in the begining
            'conv_hidden_layers': [1, 2, 3],
            'conv_depth_shape': ['brick'],
            'conv_size_shape': ['brick'],
            'conv_depth_first_neuron': [20, 40, 60],
            'conv_depth_last_neuron': [20, 40, 60],
            'conv_size_first_neuron': [5, 7],
            'conv_size_last_neuron': [3, 5],
            # fully connected layers at the end
            'first_neuron': [32, 64],
            'last_neuron': [32, 64],
            'shapes': ['brick'],
            'hidden_layers': [1, 2, 3],
            'dropout': [0.05],
            'input_shape':[input_shape],
            'verbose':[0,1]
        }
    return params
def run_custom_model(my_training_batch_generator,my_val_batch_generator,input_shape):
    params = set_params(input_shape)
    verbose = True
    round_limit = 2  # NOTE Set this to however many rounds you want to test with

    model = custom_model.create_model(my_training_batch_generator, my_val_batch_generator, verbose)

    dummyX, dummyY = my_training_batch_generator.__getitem__(0)
    testX, testY = my_val_batch_generator.__getitem__(0)
    my_val_batch_generator.on_epoch_end()
    tt = talos.Scan(x=dummyX
                        , y=dummyY
                        , params=params
                        , model=model
                        , x_val=testX
                        , y_val=testY
                        , experiment_name='example.csv'
                        , print_params=True
                        , round_limit=round_limit
                        )

    # print(vars(tt), dir(tt))
    # print(tt)
    t = project_object(tt, 'params', 'saved_models', 'saved_weights', 'data', 'details', 'round_history')
    save_object(t, 'example.pickle')


def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, protocol=2)


def load_object(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def project_object(obj, *attributes):
    out = {}
    for a in attributes:
        out[a] = getattr(obj, a)
    return out


def print_hyperparameter_search_stats(t):
    print('print_hyperparameter_search_stats: ')
    print(" *** params: ",
              {p: (v if len(v) < 200 else [v[0], v[1], v[2], '...', v[-1]]) for p, v in t['params'].items()})
    print()
        # print(" *** peak_epochs_df ",type(t['peak_epochs_df']),len(t['peak_epochs_df'].index))
        # print(t['peak_epochs_df'].to_string())
        # print()
    print(" *** data ", type(t['data']), len(t['data']))
    print(t['data'].sort_values('accuracy', ascending=False).to_string())
    print()
    distinct_data = t['data']
    nunique = distinct_data.apply(pd.Series.nunique)
    cols_to_drop = nunique[nunique == 1].index
    distinct_data = distinct_data.drop(cols_to_drop, axis=1)
    print(nunique, cols_to_drop)
    print(" *** distinct data ", type(distinct_data), len(distinct_data))
    print(distinct_data.sort_values('accuracy', ascending=False).to_string())
    print()
    print(" *** details ", type(t['details']), len(t['details']))
    print(t['details'])
    print()


# tt = load_object('example.pickle')
# print_hyperparameter_search_stats(tt)


def print_eval_table(talos_parameter_eval):
    print('print_eval_table: ')
    print(talos_parameter_eval['details'])
    for ttt in talos_parameter_eval['round_history']:
        table = PrettyTable()
        table.set_style(PLAIN_COLUMNS)
        iterations = max([len(x) for x in ttt.values()])
        table.add_column('epoch', range(1, iterations + 1))
        for key, val in sorted(ttt.items()):
            table.add_column(key, sorted(val))

        print(table)




 # p = {'first_neuron': [9, 10, 11],
    #      'hidden_layers': [0, 1, 2],
    #      'batch_size': [30],
    #      'epochs': [100],
    #      'dropout': [0],
    #      'kernel_initializer': ['uniform', 'normal'],
    #      'optimizer': ['Adam'],
    #      'losses': ['binary_crossentropy'],
    #      'activation': ['relu'],
    #      'last_activation': ['sigmoid']}




    # p = {'lr': (0.1, 10, 10),
    #      'first_neuron': [4, 8, 16, 32, 64, 128],
    #      'batch_size': [8, 16, 32],
    #      'epochs': [1, 2, 3],
    #      'dropout': (0, 0.40, 10),
    #      'optimizer': [Adam],
    #      'loss': ['categorical_crossentropy'],
    #      'activation':['relu'],
    #      'kernel_initializer': ['uniform', 'normal'],
    #      'last_activation': ['softmax'],
    #      'weight_regulizer': [None]}
    # talos_scan = talos.Scan(
    #     x=X_train_filenames,y=y_train,
    #     x_val=X_val_filenames, y_val=y_val,
    #     model=kerasModel.custom_net,
    #     params=p,
    #     experiment_name='test',
    #     round_limit=2
    # )
