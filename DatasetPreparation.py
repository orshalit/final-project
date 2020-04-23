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

move_dir = False
saveData = True

'''moves all images to all_images'''
if move_dir == True:# TODO: make this part generic, move all sub dir images to parent class dir and then delete sub dirs
    for subdir, dirs, files in os.walk(train_dir):
        # print(files)
        for file in files:
            full_path = os.path.join(subdir, file)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            parent_dir = os.path.dirname(subdir)
            if parent_dir != train_dir:
                shutil.copy2(full_path,parent_dir)
            # print('full path: ',full_path)
            # print('dest dir: ',dest_dir)
            # print('sub dir: ',subdir)
            # print('parent dir: ',parent_dir)
            # print('dirs: ',dirs)
            # print('file: ',file)
            shutil.copy2(full_path, dest_dir)
            counter = counter + 1
    print('Moved %s files to %s '% (counter,dest_dir))

#
# exit(1)
# TODO:maybe make a function out of this?

if saveData == True:
    subdirs, dirs, files = os.walk(dest_dir).__next__()
    m = len(files)
    print(m)

    filenames = []
    labels = np.zeros((m, 1))


    images_dir = dest_dir
    filenames_counter = 0
    labels_counter = -1
    print('len of filenames before saving: ',len(filenames))
    print('shape of labels before saving: ',labels.shape)

    for subdir, dirs, files in os.walk(train_dir):
        print(files)
        for file in files:
            if file.endswith('png'):
                filenames.append(file)
                # print('f: ',filenames_counter)
                # print('l: ',labels_counter)
                labels[filenames_counter, 0] = labels_counter
                filenames_counter = filenames_counter + 1
        labels_counter = labels_counter + 1

    print('len of filenames: ',len(filenames))
    print('shape of labels: ',labels.shape)


    # saving the filename array as .npy file
    print('Saving filenames and labels as .npy')
    np.save('filenames.npy', filenames)




    # One hot vector representation of labels
    y_labels_one_hot = to_categorical(labels)

    # saving the y_labels_one_hot array as a .npy file
    np.save('labels.npy', y_labels_one_hot)

    filenames = np.load('filenames.npy')
    labels = np.load('labels.npy')

    filenames_shuffled, labels_shuffled = shuffle(filenames, labels)

    # saving the shuffled file.
    # you can load them later using np.load().
    np.save('labels_shuffled.npy', labels_shuffled)
    np.save('filenames_shuffled.npy', filenames_shuffled)



    # # this is how you load
    # with open('label_to_integer.json') as a:
    #     a = json.load(a)
    #
    # with open('integer_to_label.json') as b:
    #     b = json.load(b)
    filenames_shuffled = np.load('filenames_shuffled.npy')
    labels_shuffled = np.load('labels_shuffled.npy')


    print('image: ',filenames[100])
    print('image 1hot label: ',labels[100])
    print('image integer label: ',np.argmax(labels[100]))
    print(len(filenames))
    print(len(labels))

    filenames_shuffled_numpy = np.array(filenames_shuffled)
    #70% train, 25% test, 5% validation
    X_train_filenames, X_filenames, y_train, y_test1 = train_test_split(
        filenames_shuffled_numpy, labels_shuffled, test_size=0.3, random_state=1)

    X_test_filenames, X_val_filenames, y_test, y_val = train_test_split(
        X_filenames, y_test1, test_size=(float(1/6)), random_state=1)

    print('X_train_filenames.shape: ',X_train_filenames.shape) # (3800,) example
    print('y_train.shape: ',y_train.shape)           # (3800, 12) example

    print('X_test_filenames.shape: ',X_test_filenames.shape)   # (950,) example
    print('y_test.shape: ',y_test.shape)             # (950, 12) example

    print('X_val_filenames.shape: ',X_val_filenames.shape)   # (950,) example
    print('y_val.shape: ',y_val.shape)

    # You can save these files as well. As you will be using them later for training and validation of your model.
    np.save('X_train_filenames.npy', X_train_filenames)
    np.save('y_train.npy', y_train)

    np.save('X_test_filenames.npy', X_test_filenames)
    np.save('y_test.npy', y_test)

    np.save('X_val_filenames.npy', X_val_filenames)
    np.save('y_val.npy', y_val)

    labels = images.getLabels()
    label_to_integer = {}
    integer_to_label = {}
    for index, value in enumerate(labels):    #makes class dict for every folder
        label_to_integer[value] = index
        integer_to_label[index] = value
        index = index + 1



    #save dict of label to integer and integer to label as json for later use

    with open('label_to_integer.json', 'w') as fp:
        json.dump(label_to_integer, fp)

    with open('integer_to_label.json', 'w') as fp:
        json.dump(integer_to_label, fp)


