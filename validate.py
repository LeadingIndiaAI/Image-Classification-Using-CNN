# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 09:42:56 2018

@author: leksh
"""
from keras import backend as k
from keras.models import load_model
from keras import Model

from keras.preprocessing.image import ImageDataGenerator

test_datagen = ImageDataGenerator(rescale=1./255,
                                   horizontal_flip=True,
                                   vertical_flip =True,
                                   fill_mode="nearest",
                                   zoom_range=0.3,
                                   width_shift_range=0.3,
                                   height_shift_range=0.3,
                                   channel_shift_range=0.3,
                                   rotation_range=30)
 
training_set = test_datagen.flow_from_directory('test',
                                                target_size = (224, 224),
                                                batch_size = 20,
                                                class_mode = 'categorical')
"""model_final = load_weights('test1.h5')"""

model_final.load_weights('test7.h5', by_name=True)
print('     DIBR         NATURAL      RETARGETTED   SCREENSHOTS  ')

training_set.class_indices

X,y = training_set.next()
result = model_final.predict(X)
print(result)