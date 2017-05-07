import keras
from keras import optimizers
from keras.models import Model, load_model
import keras.losses
from keras.preprocessing import image as image_utils
from keras.applications.vgg16 import preprocess_input
from keras.layers import  Input, Conv2D
from PIL import Image
from keras import backend as K
import scipy
from scipy.ndimage.filters import gaussian_filter

import numpy
def peak_signal_to_noise_ratio(true, predicted):
    return -10.0 * K.log(1.0 / (K.mean(K.square(predicted - true)))) / K.log(10.0)

file_path = 'C:/Users/xenyb/OneDrive/Documents/faces/image_0120.jpg'
file_path = 'C:/Users/Mike Wang/Downloads/Set14_SR/Set14/image_SRF_2/img_001_SRF_2_LR.png'

image = Image.open(file_path)
image_array = numpy.array(image)
image = scipy.ndimage.gaussian_filter(image_array, sigma = .5)
image = Image.fromarray(image)
image = image.resize((224,148))
image = image.resize((896,592), Image.BICUBIC)
intermediate_image = image
image = numpy.array(image)
image = image.astype('float32')
image /= 255
X_list = []
X_list.append(image)
X_list = numpy.array(X_list)
init = Input(shape=(592,896,3))
x = Conv2D(64,(9,9),activation='relu', padding='same')(init)
x = Conv2D(32,(1,1),activation='relu', padding='same')(x)
out = Conv2D(3, (5,5), padding='same')(x)
model = Model(init, out)
model.compile(loss='mse',
              optimizer=keras.optimizers.Adadelta(),
              metrics=[peak_signal_to_noise_ratio])
model.load_weights('trained_model.h5')
result = model.predict(X_list, batch_size = 1)
result *= 255
result = result[0]
result = result.astype('uint8')
image = Image.fromarray(result)
image.show()
intermediate_image.show()



