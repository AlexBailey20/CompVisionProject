from __future__ import print_function
import keras
import numpy
import scipy
from scipy.ndimage.filters import gaussian_filter
import os
from PIL import Image
import glob
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv2D
from keras import backend as K

batch_size = 5

def peak_signal_to_noise_ratio(true, predicted):
    return -10.0 * K.log(1.0 / (K.mean(K.square(predicted - true)))) / K.log(10.0)
def get_inputs():
    file_path = 'C:/Users/xenyb/OneDrive/Documents/faces/*.jpg'
    X = []
    Y = []
    for file in glob.glob(file_path):
        image=Image.open(file)
        image_y = scipy.ndimage.gaussian_filter(image, sigma = 0.5)
        image_y = image.resize((224,148))
        image_y = image_y.resize((896,592), Image.BICUBIC)
        X.append(numpy.array(image))
        Y.append(numpy.array(image_y))
    X = numpy.array(X)
    Y = numpy.array(Y)
    X=X.astype('float32')
    Y=Y.astype('float32')
    X /= 255
    Y /= 255    
    return X,Y
        
Y_train,X_train = get_inputs()
init = Input(shape=(592,896,3))
x = Conv2D(64,(9,9),activation='relu', padding='same')(init)
x = Conv2D(32,(1,1),activation='relu', padding='same')(x)
out = Conv2D(3, (5,5), padding='same')(x)
model = Model(init, out)
model.compile(loss='mse',
              optimizer=keras.optimizers.Adadelta(),
              metrics=[peak_signal_to_noise_ratio])
model.fit(x=X_train, y = Y_train, batch_size=batch_size, epochs=1, verbose=1)