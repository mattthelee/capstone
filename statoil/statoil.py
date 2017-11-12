#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 17:05:40 2017

@author: leem
"""

# need to find a sensible way to import the data. 
# Would like to handle the data in something like a datframe 
# am i able to load all the data into memory

import json
import pandas as pd
import keras
import numpy as np
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential

def loadData(filepath):
    with open(filepath) as Json:
        data = pd.DataFrame(json.load(Json))
    print "Data loaded from", filepath
    return data

def convertBand(band):
    band_array = np.array(band)
    band_array.shape = (75,75)
    return np.asarray([band_array])

def convertBands(band1,band2):
    band1_array = np.array(band1)
    band1_array.shape = (75,75)
    band2_array = np.array(band1)
    band2_array.shape = (75,75)
    return np.asarray([band1_array,band2_array])

def convertToTensor(dataframe):
    tensor = map(lambda a,b : convertBands(a,b), training_frame['band_1'], training_frame['band_2'])
    return np.asarray(tensor)

training_frame = loadData("data/processed/train.json")
# have to convert y_train to a numpy array as dataframe has a keras bug
y_train = np.asarray(pd.get_dummies(training_frame['is_iceberg']))
#TODO need to add channel into the band_1_arrays array
band_1_arrays = np.asarray(map(lambda v : convertBand(v), training_frame['band_1']))
band_2_arrays = np.asarray(map(lambda v : convertBand(v), training_frame['band_2']))
channelled_bands = convertToTensor(training_frame)
# Start neural network
#TODO need to work out how to get the conv to consider each band as a channel rather than as two separate images. 
# Als0 need to check that having it as a channel isn't just something done for different colours as the polarization is a bit different
#TODO need to pass into a conv as an array not a list. np.array(training_frame['band_1'][x]).shape = (75,75)

def createModel():
    
    cnnModel = Sequential()
    #If doing stuff with single channel
    #cnnModel.add(Conv2D(64,(3,3), strides=2, input_shape=(1,75,75), data_format='channels_first'))
    cnnModel.add(Conv2D(64,(3,3), strides=2, input_shape=(2,75,75), data_format='channels_first'))
    cnnModel.add(Conv2D(32,(2,2),data_format='channels_first'))
    cnnModel.add(MaxPooling2D(pool_size=(2,2),data_format='channels_first'))
    cnnModel.add(Flatten())
    cnnModel.add(Dense(64,activation='relu'))
    cnnModel.add(Dense(2, activation='sigmoid')) 
    
    cnnModel.summary()
    cnnModel.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return cnnModel
    
cnnModel = createModel()
cnnModel.fit(channelled_bands,y_train, epochs=10, batch_size=20)
