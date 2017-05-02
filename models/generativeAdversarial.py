import glob
import numpy as np
import math

from PIL import Image
from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Flatten, Dense, Reshape, Activation
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D

##TODO
##Make GAN resolution independant (more convolution math!)
##Decode output of generator as image

class GAN_ISR_Model(object):
    #TODO Reference http://arxiv.org/abs/1511.06434 and implement proposed changes
    # these include different activations for generation and different pooling/convolution operations
    def generator_model(self):
        #process random seed vector
        model = Sequential()
        model.add(Dense(1024, input_shape=(self.noise_shape,), activation='tanh'))
        model.add(Dense(3*100*100))
        ##unsure of the point of this -- BatchNormalization brings mean to 0 and
        ##stddev to 1 -- which centers data around tanh activation?
        #model.add(BatchNormalization())
        #model.add(Activation('tanh'))
        #upscale image
        model.add(Reshape((100, 100, 3)))
        model.add(Conv2DTranspose(64, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='tanh'))
        model.add(Conv2DTranspose(3, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='tanh'))
        return model

    def discriminator_model(self):
        input_shape = (400, 400, 3) #tuple(list(self.orig_image_shape)[-1::-1]) #(1, 28, 28)
        model = Sequential()
        model.add(Conv2D(128, kernel_size=(5,5), strides=(2,2), padding='same', input_shape=input_shape, activation='tanh'))
        model.add(Conv2D(64, kernel_size=(5,5), strides=(2,2), activation='tanh'))
        model.add(Flatten())
        #model.add(Dense(256, activation='tanh'))
        model.add(Dense(1, activation='sigmoid'))
        return model

    @staticmethod
    def combine_images(generated_images):
        num = generated_images.shape[0]
        width = 400
        height = 400
        shape = generated_images.shape[2:]
        image = np.zeros((height*shape[0], width*shape[1]),dtype=generated_images.dtype)
        for index, img in enumerate(generated_images):
            i = int(index/width)
            j = index % width
            image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = img[0, :, :]
        return image

    @staticmethod
    def gen_dis_model(generator, discriminator):
        model = Sequential()
        model.add(generator)
        discriminator.trainable = False
        model.add(discriminator)
        return model

    def train(self):
        generator = self.generator_model()
        discriminator = self.discriminator_model()
        generator_discriminator = GAN_ISR_Model.gen_dis_model(generator, discriminator)
        g_optimizer = SGD(lr=0.0005, momentum=0.9, nesterov=True)
        d_optimizer = SGD(lr=0.0005, momentum=0.9, nesterov=True)
        generator.compile(loss='binary_crossentropy', optimizer="SGD")
        generator_discriminator.compile(loss='binary_crossentropy', optimizer=g_optimizer)
        #we only want to change our discriminator's weights at certain points
        discriminator.trainable = True
        discriminator.compile(loss='binary_crossentropy', optimizer=d_optimizer)

        noise = np.zeros((self.batch_size, self.noise_shape))
        num_batches = int(len(self.orig_images)/self.batch_size)
        for epoch in range(self.epochs):
            print("Epoch: {}, Number of batches: {}".format(epoch, num_batches))
            for batch in range(num_batches):
                #make some noise, use that to make some images
                for i in range(self.batch_size):
                    noise[i, :] = np.random.uniform(-1, 1, self.noise_shape)
                image_batch = self.orig_images[batch*self.batch_size:(batch+1)*self.batch_size]
                generated_images = generator.predict(noise, verbose=0)
                #save a few images so we can track progress DEBUG

                if batch % 20 == 0:
                    image = GAN_ISR_Model.combine_images(generated_images)
                    image = image*127.5+127.5
                    Image.fromarray(image.astype(np.uint8)).save("generated_image.png")
                #teach discriminator what real images look like compared to the fakes we made
                X = np.concatenate((image_batch, generated_images))
                y = [1]*self.batch_size + [0]*self.batch_size
                discriminator_loss = discriminator.train_on_batch(X, y)
                #make some noise, use that to train our conjoined model to generate better fakes
                for i in range(self.batch_size):
                    noise[i, :] = np.random.uniform(-1, 1, self.noise_shape)
                discriminator.trainable = False
                generator_loss = generator_discriminator.train_on_batch(noise, [1]*self.batch_size)
                discriminator.trainable = True
                ##
                if batch % 10 == 9:
                    generator.save_weights('generator', True)
                    discriminator.save_weights('discriminator', True)

    def generate(self, with_discriminator=False):
        generator = self.generator_model()
        generator.compile(loss='binary_crossentropy', optimizer='SGD')
        generator.load_weights('generator')
        if with_discriminator:
            #TODO
            pass
        else:
            noise = np.zeros((self.batch_size, self.noise_shape))
            for i in range(self.batch_size):
                noise[i, :] = np.random.uniform(-1, 1, self.noise_shape)
            generated_images = generator.predict(noise, verbose=1)
            image = GAN_ISR_Model.combine_images(generated_images)
        #https://github.com/jacobgil/keras-dcgan/blob/master/dcgan.py
        image = image*127.5+127.5
        Image.fromarray(image.astype(np.uint8)).save("generated_image.png")


    def __init__(self, orig, small, batch_size=5, epochs=5):
        self.orig_images = orig
        self.small_images = small
        print("Loaded {} images into GAN ISR model".format(len(orig)))
        #shape = (batch, height, weight, channels) ##(data_format='channels_last')
        self.orig_image_shape = (400,400)
        self.small_image_shape = (100,100)
        self.noise_shape = 100
        self.batch_size = batch_size
        self.epochs = epochs
