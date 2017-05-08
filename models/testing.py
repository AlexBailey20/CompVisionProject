import keras
from keras import optimizers
from keras.models import Model, load_model
import keras.losses
from keras.preprocessing import image as image_utils
from keras.applications.vgg16 import preprocess_input
from keras.layers.normalization import BatchNormalization
from keras.layers import  Input, Conv2D, UpSampling2D, Activation
from PIL import Image
from keras import backend as K
from keras.layers.merge import add
import scipy
import matplotlib.pyplot as plt

from scipy.ndimage.filters import gaussian_filter
import numpy
def bn_relu(input):
    norm = BatchNormalization(axis=3)(input)
    return Activation("relu")(norm)

def bn_relu_conv(filters, kernel_size):
    def f(input):
        activation = bn_relu(input)
        return Conv2D(filters=filters, kernel_size=kernel_size, padding='same')(activation)
    return f

def bottleneck(filters):
    def f(input):
        conv_1_1 = bn_relu_conv(filters=filters, kernel_size=(3, 3))(input)
        conv_3_3 = bn_relu_conv(filters=filters, kernel_size=(3, 3))(conv_1_1)
        residual = bn_relu_conv(filters=filters, kernel_size=(1, 1))(conv_3_3)
        return add([input, residual])
    return f
def peak_signal_to_noise_ratio(true, predicted):
    return -10.0 * K.log(1.0 / (K.mean(K.square(predicted - true)))) / K.log(10.0)

file_path = 'C:/Users/xenyb/OneDrive/Documents/jpg/image_0072.jpg'
file_path = 'C:/Users/Mike Wang/Downloads/faces/image_0446.jpg'
#file_path = 'C:/Users/Mike Wang/Downloads/Set14_SR/Set14/image_SRF_2/img_001_SRF_2_LR.png'

image = Image.open(file_path)
image_array = numpy.array(image)
image = scipy.ndimage.gaussian_filter(image_array, sigma = .5)
#cv2.imshow('denoise',image)
plt.imshow(image)
image = Image.fromarray(image)
image = image.resize((64,64))
intermediate_image = image.resize((256,256), Image.BICUBIC)
intermediate_image.show()
image = numpy.array(image)
image = image.astype('float32')
image /= 255
X_list = []
X_list.append(image)
X_list = numpy.array(X_list)
init = Input(shape=(64,64,3))
out = Conv2D(64,(3,3),activation='relu', padding='same')(init)
out = bottleneck(filters=64)(out)
out = bottleneck(filters=64)(out)
out = bottleneck(filters=64)(out)
out = bottleneck(filters=64)(out)
out = bottleneck(filters=64)(out)
out = bottleneck(filters=64)(out)
out = bottleneck(filters=64)(out)
out = bottleneck(filters=64)(out)
out = bottleneck(filters=64)(out)
out = bottleneck(filters=64)(out)
out = UpSampling2D(2)(out)
out = Conv2D(64,(3,3),activation='relu', padding='same')(out)
out = UpSampling2D(2)(out)
out = Conv2D(64,(3,3),activation='relu', padding='same')(out)
out = Conv2D(64,(3,3),activation='relu', padding='same')(out)
out = Conv2D(3,(3,3), padding='same')(out)
model = Model(init, out)
model.compile(loss='mse',
              optimizer=keras.optimizers.Adadelta(),
              metrics=[peak_signal_to_noise_ratio])
model.load_weights('trained_residual_model.h5')
result = model.predict(X_list, batch_size = 1)
result = result[0]
est = numpy.array(intermediate_image)
est = est.astype('float32')
result *= 255
result = result + est
result /= 1.25
result = numpy.clip(result, a_min=0, a_max=255)
result = result.astype('uint8')
image = Image.fromarray(result)
image.show()
