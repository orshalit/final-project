import os
import shutil
import numpy as np
from keras.utils import to_categorical
from sklearn.utils import shuffle
import json
import imageAugmentor as images
from sklearn.model_selection import train_test_split
import glob

train_dir = "C:/Users/אור/Desktop/Pearls-Images/train"                   #enter train folder path
dest_dir = "C:/Users/אור/Desktop/Pearls-Images/all_images"               #enter all_images folder path (may need to create an empty folder called all_images)
counter = 0
n_samples = 10 #number of samples for to augment each class
"""convert and/or augment"""
convert = False #change to True to activate
augment = False #change to True to activate
"""orginize folders and images"""
move_dir = False    #change to True to activate
"""preprocess images and create dicts and lists of items"""
saveData = False    #change to True to activate



''' changes all the names in the dataset to numbers, 
    if there is a problem changing again then change the prefix 'image' in os.rename to be something else'''
def shorten_file_names():
    number = 1
    for subdir, dirs, files in os.walk(train_dir):
        for file in files:
            full_path = os.path.join(subdir, file)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".png"):
                # dir_path = os.path.dirname(os.path.abspath(full_path))
                try:
                    os.rename(full_path, subdir + '/' +'image' + "{0}".format(number) + '.png')
                except OSError as e:
                    print("Something happened, cannot change filename:", e)
                number += 1


images.root_directory= train_dir+'/*'
images.images_directory = train_dir+'/*/*'
if convert == True:
    images.convert_to_png()
if augment == True:
    images.augment_images(n_samples)

'''moves all images to all_images'''
if move_dir == True:# TODO: make this part generic, move all sub dir images to parent class dir and then delete sub dirs
    for subdir, dirs, files in os.walk(train_dir):
        print(subdir)
        for file in files:
            full_path = os.path.join(subdir, file)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            parent_dir = os.path.dirname(subdir)
            if parent_dir != train_dir:
                shutil.copy2(full_path,parent_dir)
                os.remove(full_path)
            """check if sub dir is empty and delete"""
            if not os.listdir(subdir):
                os.rmdir(subdir)
    shorten_file_names()
    for subdir, dirs, files in os.walk(train_dir):
        for file in files:
            full_path = os.path.join(subdir, file)
            shutil.copy2(full_path, dest_dir)
            counter = counter + 1
    print('Moved %s files to %s '% (counter,dest_dir))

#
# exit(1)
# TODO:maybe make a function out of this?

if saveData == True:
    subdirs, dirs, files = os.walk(dest_dir).__next__()
    m = len(files)
    print('total amount of images: ',m)

    filenames = []
    labels = np.zeros((m, 1))


    images_dir = dest_dir
    filenames_counter = 0
    labels_counter = -1
    labels_list = []
    integer_to_label = {}
    label_to_integer = {}
    print('len of filenames before saving: ',len(filenames))
    print('shape of labels before saving: ',labels.shape)

    for subdir, dirs, files in os.walk(train_dir):
        print(subdir.split(os.path.sep)[-1])
        currentdir = subdir.split(os.path.sep)[-1]
        for file in files:

            if file.endswith('png'):
                #if labels_counter == 2 or labels_counter == 3:
                filenames.append(file)
                # print('f: ',filenames_counter)
                # print('l: ',labels_counter)
                labels[filenames_counter, 0] = labels_counter
                filenames_counter = filenames_counter + 1
        if currentdir != train_dir:
            integer_to_label[labels_counter] = currentdir
            label_to_integer[currentdir] = labels_counter
            labels_list.append(currentdir)
        labels_counter = labels_counter + 1
    labels =  labels[:len(filenames)]
    print('int to label: ',integer_to_label)
    print('label to int: ',label_to_integer)
    # print('labels: ',labels[570])
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

    #save dict of label to integer and integer to label as json for later use

    with open('label_to_integer.json', 'w') as fp:
        json.dump(label_to_integer, fp)

    with open('integer_to_label.json', 'w') as fp:
        json.dump(integer_to_label, fp)

    with open('labels_list.json', 'w') as fp:
        json.dump(labels_list, fp)


