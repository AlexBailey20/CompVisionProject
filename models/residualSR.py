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

def get_inputs():
    file_path = 'C:/Users/xenyb/OneDrive/Documents/jpg/*.jpg'
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
model.compile(loss='mse',
              optimizer=keras.optimizers.Adadelta())
model.fit(x=X_train, y = Y_train, batch_size=10, epochs=10, verbose=1)
model.save('trained_residual_model.h5')