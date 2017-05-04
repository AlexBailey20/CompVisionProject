import numpy as np
import math
import matplotlib.pyplot as plt

from PIL import Image
from keras import backend as tf
from keras.models import Sequential
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Flatten, Dense, Reshape, Activation, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, Conv2DTranspose, UpSampling2D

class GAN_ISR_Model(object):
    #inspired mostly by https://medium.com/towards-data-science/gan-by-example-using-keras-on-tensorflow-backend-1a6d515a60d0
    def generator_model(self):
        #Input: 192 floats
        #   :   8x8x3 floats
        #Output: 256x256x3 floats
        model = Sequential()
        model.add(Dense(8*8*3, input_shape=self.generator_input_shape))
        model.add(BatchNormalization(momentum=self.momentum))
        model.add(Activation('relu'))
        model.add(Reshape((8, 8, 3)))
        model.add(Dropout(self.dropout))
        model.add(UpSampling2D(2))
        model.add(Conv2DTranspose(512, kernel_size=4, padding='same', kernel_initializer='glorot_normal'))
        model.add(BatchNormalization(momentum=self.momentum))
        model.add(Activation('relu'))
        model.add(UpSampling2D(4))
        model.add(Conv2DTranspose(256, kernel_size=5, padding='same', kernel_initializer='glorot_normal'))
        model.add(BatchNormalization(momentum=self.momentum))
        model.add(Activation('relu'))
        model.add(UpSampling2D(2))
        model.add(Conv2DTranspose(128, kernel_size=4, padding='same', kernel_initializer='glorot_normal'))
        model.add(BatchNormalization(momentum=self.momentum))
        model.add(Activation('relu'))
        model.add(UpSampling2D(2))
        #should you use strided convolutions here instead? need to investigate
        model.add(Conv2DTranspose(3, kernel_size=4, padding='same', kernel_initializer='glorot_normal'))
        model.add(Activation('sigmoid'))
        return model

    def discriminator_model(self):
        #Input: 256x256x3 floats
        #Output: {0,1}
        model = Sequential()
        model.add(Conv2D(32, kernel_size=5, strides=4, padding='same', input_shape=self.discriminator_input_shape))
        model.add(LeakyReLU(self.relu_alpha))
        model.add(Dropout(self.dropout))
        model.add(Conv2D(64, kernel_size=5, strides=4, padding='same'))
        model.add(LeakyReLU(self.relu_alpha))
        model.add(Dropout(self.dropout))
        model.add(Conv2D(128, kernel_size=4, strides=2, padding='same'))
        model.add(LeakyReLU(self.relu_alpha))
        model.add(Dropout(self.dropout))
        model.add(Conv2D(256, kernel_size=3, strides=2, padding='same'))
        model.add(LeakyReLU(self.relu_alpha))
        model.add(Dropout(self.dropout))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))
        return model

    @staticmethod
    def gen_dis_model(generator, discriminator):
        #Input: 192 floats
        #Output: {0,1}
        model = Sequential()
        model.add(generator)
        model.add(discriminator)
        return model

    def random_noise(self):
        #create just the right amount of random noise for generation
        return np.random.uniform(-1, 1, self.noise_shape)

    def train(self):
        #TODO better saving image behavior; need to keep track of past epochs
        def save_generated_images(gen_images):
            for num, image in enumerate(gen_images):
                plt.imshow(image)
                plt.savefig("generated_image_{}.png".format(num))
        #instantiate models
        generator = self.generator_model()
        discriminator = self.discriminator_model()
        adversarial = GAN_ISR_Model.gen_dis_model(generator, discriminator)
        #assign loss and optimizers
        discriminator.compile(loss='binary_crossentropy', optimizer=self.dis_opt)
        adversarial.compile(loss='binary_crossentropy', optimizer=self.gen_opt)
        #pre-train discriminator
        image_batch = self.images[:self.batch_size]
        generated_images = generator.predict(self.random_noise(), verbose=0)
        X = np.concatenate((image_batch, generated_images))
        y = np.zeros([self.batch_size*2])
        y[:self.batch_size] = 1
        discriminator.fit(X, y, batch_size=self.batch_size, verbose=0)
        y_hat = discriminator.predict(X)
        #train for multiple epochs
        def train(epochs=5):
            for e in range(epochs+1):
                #collect real and generated images
                image_batch = self.images[np.random.randint(0, self.images.shape[0], size=self.batch_size),:,:,:]
                generated_images = generator.predict(self.random_noise(), verbose=0)
                X = np.concatenate((image_batch, generated_images))
                y = np.ones([2*self.batch_size])
                y[self.batch_size:] = 0
                #train discriminator to recognize which images are generated
                d_loss = discriminator.train_on_batch(X, y)
                #train generator to fool discriminator on generated images
                y = np.ones([self.batch_size])
                g_loss = adversarial.train_on_batch(self.random_noise(), y)
                #print losses and save weights/images
                if e % 5 == 0:
                    print("Epoch\t{}\t\tD:{}\t\tG:{}".format(e, d_loss, g_loss))
                    if e % 10 == 0:
                        save_generated_images(generated_images)
                        generator.save_weights('generator.weights', True)
                        discriminator.save_weights('discriminator.weights', True)
        train(self.epochs)

    def generate(self, with_discriminator=False):
        #TODO write generation code
        generator = self.generator_model()
        generator.compile(loss='binary_crossentropy', optimizer=self.gen_opt)
        generator.load_weights('generator.weights')
        if with_discriminator:
            #TODO
            pass
        else:
            generated_images = generator.predict(self.random_noise(), verbose=1)
            img = generated_images[0]
            #TODO
            pass


    def __init__(self, images, batch_size=20, epochs=30, dropout=0.3, relu_alpha=0.2, momentum=0.9):
        #TODO clean up, simplify where possible
        self.images = np.asarray(images)
        print("Loaded {} images into GAN ISR model".format(len(self.images)))
        self.random_values = 100
        self.batch_size = batch_size
        self.generator_input_shape = (self.random_values,)
        self.discriminator_input_shape = (self.images[0].shape)
        self.noise_shape = (self.batch_size, self.random_values)

        #tunable model parameters
        self.epochs = epochs
        self.gen_opt = RMSprop(lr=0.0004, clipvalue=1.0, decay=3e-8)
        self.dis_opt = RMSprop(lr=0.0008, clipvalue=1.0, decay=6e-8)
        self.dropout = dropout
        self.relu_alpha = relu_alpha
        self.momentum = momentum
