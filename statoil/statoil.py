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
from keras.layers import Dropout, Flatten, Dense, Input
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
import png


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

def createVGGModel():
    anglesInput = Input(shape=[1], name ="angles")
    angles_layer = Dense(1,)(anglesInput)
    
    transferred_model = VGG16(weights='imagenet')

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

def runModel(channelled_bands, y_train):
    cnnModel = createModel()
    return cnnModel.fit(channelled_bands,y_train, epochs=10, batch_size=20)

def showArrayImage(array):
    flat_array = array.flatten()
    new_array = np.asarray(map(lambda v: int(v + 40), flat_array))
    new_array.shape = (75,75)
    arrayImage = Image.fromarray(new_array)
    return arrayImage.show()
    
training_frame = loadData("data/processed/train.json")
testing_frame = loadData("data/processed/test.json")
# have to convert y_train to a numpy array as dataframe has a keras bug
y_train = np.asarray(pd.get_dummies(training_frame['is_iceberg']))
#TODO need to add channel into the band_1_arrays array
band_1_arrays = np.asarray(map(lambda v : convertBand(v), training_frame['band_1']))
band_2_arrays = np.asarray(map(lambda v : convertBand(v), training_frame['band_2']))
channelled_bands = convertToTensor(training_frame)

# TODO experiment without horizontal, vertical and rotations
# Sun direction might affect the light see so might not want to lose that information
gen = ImageDataGenerator(horizontal_flip = True,
                         vertical_flip = True,
                         zoom_range = 0.2,
                         rotation_range = 360)
# Als0 need to check that having it as a channel isn't just something done for different colours as the polarization is a bit different


#immediate steps for kernal based stuff
# Need to remind myself what form the data is in when i pass it to the model
# need to convert daata into a flow and the pass into image generator
# need to pass data into the model

# Next steps:
# icebergs and ships are only a few pixels wide
# L- should therefore have the image cropped to only an area that contains the object
# L- this will reduce redundent data and speed up the algorithm see: http://elib.dlr.de/99079/2/2016_BENTES_Frost_Velotto_Tings_EUSAR_FP.pdf
# should do a 3d plot of the array too as shown in above paper
# Add in image gerneator and use it to create additional images with distortions
# use vg16 pretrained nn to help, also try the other ones available from image net
# Setup an aws instance to allow the models to be long running
# consider writing in pyspark compatible format
# Add additional conv layers
# L- research why additional conv layers would improve accuracy rather than improve performance
# could consider looking at ship wake as form of detection
# the higher the wind, the more bragg scattering, the more cluttered the ocean image will be
# the higher winds also make for improved wake detection
# clutter decreases with incidence angle

#sentinel 1 paper, can i use the following data? : At DRDC Ottawa, shore-based commercial AIS data were
#obtained in conjunction with several RADARSAT-1 and Envisat ASAR trials [32] [34] [36], for
#compilation of a database of more than 4000 validated ship signatures that may be used to
#improve models of ship RCS and its variability.

#Should consider chaning the dB into 0,255 similar to in Feature extraction of dual-pol SAR imagery for sea ice image segmentation I
# I need to find the range of the decibels,set one to 0, set the highest to 255 and apply some method to split the data across the range
# Need to check the precision of the  decibels and therefore whether this process results in loss of sig fig
    #l- from the same paper should consider perfroming the max gradient preprocessing, it's simple and was effective for them

# Should add to reading list: 


# Howell, C., Mills, J., Power, D., Youden, J., Dodge, K., Randell, C., Churchill, S., and Flett,
#D. (2006). A multivariate approach to iceberg and ship classification in HH/HV ASAR
#data. Proc. 2006 International Geoscience and Remote Sensing Symposium (IGARSS
#2006). CD-ROM proceedings. 31 July to 4 Aug. 2006, Denver, USA.


#Henschel, M.D., and Livingstone, C.E. (2006). Observation of vessel heave with airborne
#SAR. Proc. OceanSAR 2006 – The Third Workshop on Coastal and Marine Applications
#of SAR, St. John’s, NL, Canada, 23 to 25 October 2006.

#Power, D., Youden, J., Lane, K., Randell, C., and Flett, D. (2001). Iceberg detection
#capabilities of RADARSAT synthetic aperture radar. Canadian Journal of Remote Sensing,
#27(5), 476-486.

#Pond, S., and Pickard, G.L. (1983). Introductory Dynamical Oceanography, 2 nd Edition.
#Pergamon Press, Toronto.
