import keras
from skimage.transform import resize
from skimage.io import imread
import numpy as np
import DatasetPreparation

class My_Custom_Generator(keras.utils.Sequence):

    def __init__(self, image_filenames, labels, batch_size, input_shape):
        self.image_filenames = image_filenames
        self.labels = labels
        self.batch_size = batch_size
        self.input_shape = input_shape

    def __len__(self):
        return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(np.int)

    def __getitem__(self, idx):
        batch_x = self.image_filenames[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size: (idx + 1) * self.batch_size]

        return np.array([
            resize(imread(DatasetPreparation.dest_dir +'/'+ str(file_name)), self.input_shape)
            for file_name in batch_x]) / 255.0, np.array(batch_y)
    def getNumber(self):
        return len(self.labels)

    def print(self):
        print('test')