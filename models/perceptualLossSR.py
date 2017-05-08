import keras
import glob
import numpy
from PIL import Image
import scipy
from scipy.ndimage.filters import gaussian_filter
from keras.models import Model
from keras.layers import Input, Activation, Dense, Flatten
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.layers.merge import add
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras import backend as K

from keras.applications import vgg16
import h5py

def perceptual_loss(y_true, y_pred):
    vgg_model = vgg16.VGG16(include_top=False, weights='imagenet', input_shape = K.shape(y_true))
    2nd_pool = Model(inputs=vgg_model.input, outputs=vgg_model.layers[6].output)
    5th_pool = Model(inputs=vgg_model.input, outputs=vgg_model.layers[18].output)

    2true = 2nd_pool.predict(y_true)
    2pred = 2nd_pool.predict(y_pred)
    5true = 5th_pool.predict(y_true)
    5pred = 5th_pool.predict(y_pred)
    size = K.shape(2true)[1]

    loss = K.mean(K.sum(K.square(2true - 2pred))) / (size * size)
    loss += K.mean(K.sum(K.square(5true - 5pred))) / (size * size)
    return loss

def get_inputs():
    file_path = 'jpg/*.jpg'
    X = []
    Y = []
    i = 1
    for file in glob.glob(file_path):
        if(i > 70):
            break
        image=Image.open(file)
        image_y = image.resize((256,256))        
        image=image.resize((64,64))      
        X.append(numpy.array(image))
        Y.append(numpy.array(image_y))
        i += 1
    X = numpy.array(X)
    Y = numpy.array(Y)
    X=X.astype('float32')
    Y=Y.astype('float32')
    X /= 255
    Y /= 255    
    return X,Y
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

X_train,Y_train = get_inputs()
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
model.compile(loss='perceptual_loss',
              optimizer=keras.optimizers.Adadelta())
model.fit(x=X_train, y = Y_train, batch_size=10, epochs=10, verbose=1)
model.save('perceptual_loss_model.h5')
