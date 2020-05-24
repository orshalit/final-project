import Augmentor
import os
import glob
from PIL import Image
import numpy as np

root_directory = ""
images_directory =""
folders = []
samples_n=1  # how many augmented images per image


'''converting images'''
def convert_to_png():
    print('convert all images from jpeg to png and all channels to RGB')   #iterate over dir, change to png format
    for f in glob.glob(images_directory):
        # print('convert: ',f)
        if f.endswith(".jpg") or f.endswith(".jpeg"):
            im = Image.open(f)
            rgb_im = im.convert('RGB')
            rgb_im.save(f.replace("jpg", "png"), quality=100)
            os.remove(f)
        elif f.endswith(".png"):
            im = Image.open(f)
            imArray=np.asarray(im)
            if imArray.shape[2]==4:
                rgb_im = im.convert('RGB')
                rgb_im.save(f,quality=100)

    print('end of conversion')



'''augment images, samples_n means how many images to create from 1 image in the dataset'''
def augment_images(n_samples):
    pipelines = {}                             #start creating pipeline for image distortion
    for f in glob.glob(root_directory):
        if os.path.isdir(f):
            folders.append(os.path.abspath(f))
    for folder in folders:
        print("Folder %s:" % (folder))
        pipelines[os.path.split(folder)[1]] = (Augmentor.Pipeline(folder))
        print("\n----------------------------\n")

    for p in pipelines.values():
        print("Class %s has %s samples." % (p.augmentor_images[0].class_label, len(p.augmentor_images)))
        p.rotate(probability=0.5, max_left_rotation=5, max_right_rotation=5)
        p.random_distortion(probability=0.13, grid_width=1, grid_height=1, magnitude=2)  #random elastic distortion
        p.flip_random(probability=0.5)  #flip image randomly left right up down
        p.zoom_random(1, percentage_area=0.8)
        p.shear(0.1, 3, 3) #smear image
        p.rotate_random_90(0.25) #rotate
        p.crop_random(probability=0.2,percentage_area=0.7,randomise_percentage_area=False)
        p.skew(0.4, 0.5)
        p.sample(n_samples,multi_threaded=True)


