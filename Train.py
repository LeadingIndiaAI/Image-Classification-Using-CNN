# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 16:13:49 2018

@author: leksh
"""

from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as k
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
import matplotlib.pyplot as plt

# Folder structure

# data/train/3D      - 3D train samples
# data/train/Retargeted           - Retargeted train samples
# data/train/Natural           - Natural train samples
# data/train/Screenshots           - Screenshots train samples

# data/validation/3D      - 3D validation samples
# data/validation/Retargeted           - Retargeted validation samples
# data/validation/Natural			 - Natural validation samples
# data/validation/Screenshots			 - Screenshots validation samples

img_width, img_height = 139, 139        # Resolution of inputs
train_data_dir = "xtrain"           # Folder of train samples
validation_data_dir = "validation" # Folder of validation samples
nb_train_samples = 5108                # Number of train samples
nb_validation_samples = 308            # Number of validation samples
batch_size = 10                         # Batch size
epochs = 2                           # Maximum number of epochs
# Load VGG16
model=applications.VGG16(weights="imagenet", include_top=False, input_shape=(img_width, img_height, 3))
# Freeze first 15 layers
for layer in model.layers[:50]: #Total 255 layers for this model
  	layer.trainable = False
for layer in model.layers[:50]:
  	layer.trainable = True
# Attach additional layers
x = model.output
x = Flatten()(x)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.5)(x)
predictions = Dense(4, activation="softmax")(x) # 5-way softmax classifier at the end

model_final = Model(input=model.input, output=predictions)

model_final.compile(loss="categorical_crossentropy", optimizer=optimizers.SGD(lr=1e-3, momentum=0.9), metrics=["accuracy"])

# train data generator (data augmentation)
train_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True,vertical_flip =True, fill_mode="nearest", zoom_range=0.3, width_shift_range=0.3, height_shift_range=0.3,channel_shift_range=0.3, rotation_range=30)
# test data generator (data augmentation)
test_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True,vertical_flip =True, fill_mode="nearest", zoom_range=0.3, width_shift_range=0.3, height_shift_range=0.3,channel_shift_range=0.3, rotation_range=30)

# load from directory
train_generator = train_datagen.flow_from_directory(train_data_dir, target_size=(img_height, img_width), batch_size=batch_size, class_mode="categorical")
# load from directory
validation_generator = test_datagen.flow_from_directory(validation_data_dir, target_size=(img_height, img_width), class_mode="categorical")

# save models
checkpoint = ModelCheckpoint("test7.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
# early stopping
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')

# TRAINING
history = model_final.fit_generator(train_generator, samples_per_epoch=nb_train_samples, epochs=epochs, validation_data=validation_generator, nb_val_samples=nb_validation_samples, callbacks=[checkpoint, early])
model_final.save_weights("test7.h5", overwrite = 'true')
# list all data in history
print(history.history.keys())
# summarize history for train
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for test
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
