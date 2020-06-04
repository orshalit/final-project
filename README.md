# final-project
pearl type recognition

Dataset: 
https://drive.google.com/file/d/1JjLuvxLZ2RwNe5iW6sHPVEYgH5GBqRKm/view?usp=sharing


this is a simple explanation, to activate and run it requires some manual changing of parameters!!!

images should be stored in train folder under subdirs with the name of the class, and an empty folder all_images.
to run conversion to png, augmentation and preprocess run the DatasetPreparation and change the corresponding action to True
convert,augment,move_dir,saveData.
set number of samples to augment for each class with n_samples
make sure the train_dir path is set to train folder and dest_dir path set to an empty folder all_images


to build/predict model, run build-model, change the build parameter to True if you want to run a NN. 
can also change the size of the images and batch size 




this is experimental. 
