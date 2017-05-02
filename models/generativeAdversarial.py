import numpy as np
import math

from PIL import Image
from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Flatten, Dense, Reshape, Activation
from keras.layers.convolutional import Conv2D, Conv2DTranspose, UpSampling1D
from keras.layers.pooling import MaxPooling2D

class GAN_ISR_Model(object):
    #TODO Reference http://arxiv.org/abs/1511.06434 and implement proposed changes
    # these include different activations for generation and different pooling/convolution operations
    def generator_model(self):
        #Input: 48 floats
        #Output: 512x512x3 floats
        model = Sequential()
        model.add(Reshape((4, 4, 3), input_shape=self.noise_shape))
        model.add(Conv2DTranspose(32, kernel_size=6, strides=4, padding='same', activation='relu'))
        model.add(Conv2DTranspose(32, kernel_size=6, strides=4, padding='same', activation='relu'))
        model.add(Conv2DTranspose(32, kernel_size=5, strides=4, padding='same', activation='relu'))
        model.add(Conv2DTranspose(3, kernel_size=5, strides=2, padding='same', activation='relu'))
        return model

    def discriminator_model(self):
        #Input: 512x512x3 floats
        #Output: {0,1}
        model = Sequential()
        model.add(Conv2D(32, kernel_size=5, strides=4, padding='same', activation='relu', input_shape=self.image_shape))
        model.add(Conv2D(32, kernel_size=5, strides=4, padding='same', activation='relu'))
        model.add(Conv2D(32, kernel_size=5, strides=4, padding='same', activation='relu'))
        model.add(Conv2D(32, kernel_size=5, strides=2, padding='same', activation='relu'))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))
        return model

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
        generator.compile(loss='binary_crossentropy', optimizer='sgd')
        generator_discriminator.compile(loss='binary_crossentropy', optimizer=g_optimizer)
        discriminator.trainable = True
        discriminator.compile(loss='binary_crossentropy', optimizer=d_optimizer)

        noise = np.zeros((self.batch_size, self.random_values))
        def randomize(noise):
            noise = np.random.uniform(-1, 1, noise.shape)

        num_batches = int(len(self.orig_images)/self.batch_size)
        for epoch in range(self.epochs):
            print("Epoch: {}".format(epoch+1))
            for batch in range(num_batches):
                #make some noise, use that to make some images
                randomize(noise)
                image_batch = self.orig_images[batch*self.batch_size:(batch+1)*self.batch_size]
                generated_images = generator.predict(noise, verbose=0)
                #save a few images so we can track progress DEBUG
                if (batch) % self.batch_size == 0:
                    image = generated_images[0] #GAN_ISR_Model.combine_images(generated_images)
                    image = image*127.5+127.5
                    Image.fromarray(image.astype(np.uint8)).save("generated_image_{}.png".format(batch))
                #TODO rewrite loss calculation
                #teach discriminator what real images look like compared to the fakes we made
                X = np.concatenate((image_batch, generated_images))
                y = [1]*self.batch_size + [0]*self.batch_size
                discriminator_loss = discriminator.train_on_batch(X, y)
                #make some noise, use that to train our conjoined model to generate better fakes
                randomize(noise)
                discriminator.trainable = False
                generator_loss = generator_discriminator.train_on_batch(noise, [1]*self.batch_size)
                discriminator.trainable = True
                ##
                if batch % self.batch_size == 9:
                    generator.save_weights('generator.weights', True)
                    discriminator.save_weights('discriminator.weights', True)

    def generate(self, with_discriminator=False):
        #TODO write generation code
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
            pass
            # image = GAN_ISR_Model.combine_images(generated_images)
        #https://github.com/jacobgil/keras-dcgan/blob/master/dcgan.py
        #image = image*127.5+127.5
        #Image.fromarray(image.astype(np.uint8)).save("generated_image.png")


    def __init__(self, orig, small, batch_size=10, epochs=5):
        self.orig_images = orig
        self.small_images = small
        print("Loaded {} images into GAN ISR model".format(len(orig)))
        self.image_shape = orig[0].shape
        self.random_values = 48
        self.noise_shape = (self.random_values,)
        self.batch_size = batch_size
        self.epochs = epochs
