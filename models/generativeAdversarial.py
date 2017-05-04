import numpy as np
import math
import matplotlib.pyplot as plt

from PIL import Image
from keras.models import Sequential
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Flatten, Dense, Reshape, Activation, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, Conv2DTranspose, UpSampling2D

def save_generated_images(gen_images, title="generated_image", limit=None):
    #save images in list using pyplot as a wrapper (is this optimal? maybe.)
    if limit is not None:
        gen_images = gen_images[:limit]
    if len(gen_images) > 1:
        for num, image in enumerate(gen_images):
            plt.imshow(image)
            plt.savefig("{}_{}.png".format(title, num))
    else:
        plt.imshow(gen_images[0])
        plt.savefig("{}.png".format(title))

class GAN_Model(object):
    #inspired mostly by https://medium.com/towards-data-science/gan-by-example-using-keras-on-tensorflow-backend-1a6d515a60d0
    def generator_model(self):
        #Input: 100 floats
        #   :   16x16x3 floats
        #Output: 256x256x3 floats
        dense_n = 4
        model = Sequential()
        model.add(Dense(dense_n*dense_n*3, input_shape=self.generator_input_shape))
        model.add(BatchNormalization(momentum=self.momentum))
        model.add(Activation('relu'))
        model.add(Reshape((dense_n, dense_n, 3)))
        model.add(Dropout(self.dropout))
        model.add(UpSampling2D(4))
        model.add(Conv2DTranspose(512, 5, padding='same'))
        model.add(BatchNormalization(momentum=self.momentum))
        model.add(Activation('relu'))
        model.add(UpSampling2D(4))
        model.add(Conv2DTranspose(256, 5, padding='same'))
        model.add(BatchNormalization(momentum=self.momentum))
        model.add(Activation('relu'))
        model.add(UpSampling2D(2))
        model.add(Conv2DTranspose(128, 5, padding='same'))
        model.add(BatchNormalization(momentum=self.momentum))
        model.add(Activation('relu'))
        model.add(UpSampling2D(2))
        model.add(Conv2DTranspose(3, 5, padding='same'))
        model.add(Activation('sigmoid'))
        return model

    def discriminator_model(self):
        #Input: 256x256x3 floats
        #Output: {0,1}
        model = Sequential()
        model.add(Conv2D(64, 6, strides=4, padding='same', input_shape=self.discriminator_input_shape))
        model.add(LeakyReLU(self.relu_alpha))
        model.add(Dropout(self.dropout))
        model.add(Conv2D(128, 6, strides=4, padding='same'))
        model.add(LeakyReLU(self.relu_alpha))
        model.add(Dropout(self.dropout))
        model.add(Conv2D(128, 5, strides=2, padding='same'))
        model.add(LeakyReLU(self.relu_alpha))
        model.add(Dropout(self.dropout))
        model.add(Conv2D(256, 4, strides=2, padding='same'))
        model.add(LeakyReLU(self.relu_alpha))
        model.add(Dropout(self.dropout))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))
        return model

    def adversarial_model(self):
        #Input: 100 floats
        #Output: {0,1}
        model = Sequential()
        model.add(self.generator)
        model.add(self.discriminator)
        return model

    def random_noise(self, num=None):
        if num is None:
            num=self.batch_size
        return np.random.uniform(-1, 1, (num, self.random_values))

    def train(self, epochs, resume=False):
        if self.images is None:
            print("Training unavailable without loading images (remove -f).")
            return

        if resume:
            self.generator.load_weights('generator.weights', True)
            self.discriminator.load_weights('discriminator.weights', True)
            print("Loaded weights, resuming training.")

        #pre-train discriminator
        image_batch = self.images[:self.batch_size]
        generated_images = self.generator.predict_on_batch(self.random_noise())
        X = np.concatenate((image_batch, generated_images))
        y = np.zeros([self.batch_size*2])
        y[:self.batch_size] = 1
        self.discriminator.train_on_batch(X, y)

        #train for multiple epochs
        def train(epochs):
            generated_images = None #reference for saving results of each epoch
            try:
                for e in range(epochs):
                    header_text = "\t\tDiscriminator Loss\t\tGenerator Loss" if e==0 else ""
                    print("\nEpoch\t{}{}".format(e, header_text))
                    for b in range(int(self.num_images/self.batch_size)):
                        #collect real and generated images
                        image_batch = self.images[np.random.randint(0, self.num_images, size=self.batch_size)]
                        generated_images = self.generator.predict_on_batch(self.random_noise())
                        X = np.concatenate((image_batch, generated_images))
                        y = np.ones([2*self.batch_size])
                        y[self.batch_size:] = 0
                        #train discriminator to recognize which images are generated
                        d_loss = self.discriminator.train_on_batch(X, y)
                        #train generator to fool discriminator on generated images
                        y = np.ones([self.batch_size])
                        g_loss = self.adversarial.train_on_batch(self.random_noise(), y)
                        #print losses and save weights/images
                        if b % 5 == 0:
                            batch_text = "Batch\t" if b==0 else "\t"
                            print("{}{}\t\t{}\t\t{}".format(batch_text, b, d_loss, g_loss))
                    save_generated_images(generated_images, title="epoch_{}".format(e), limit=1)
            except KeyboardInterrupt:
                print("User stopped training.")
            finally:
                self.generator.save_weights('generator.weights', True)
                self.discriminator.save_weights('discriminator.weights', True)
        train(epochs)

    def generate(self, num_samples=None, with_discriminator=False):
        if num_samples is None:
            num_samples = self.batch_size
        #make some noise
        noise = self.random_noise(num=num_samples)
        #load generator weights
        self.generator.load_weights('generator.weights', True)
        if with_discriminator:
            #run this batch through the adversarial and update weights
            self.discriminator.load_weights('discriminator.weights', True)
            g_loss = self.adversarial.train_on_batch(noise, np.ones([num_samples]))
            print("Discriminator yielded loss {}".format(g_loss))
        #generate and save images
        generated_images = self.generator.predict_on_batch(noise)
        save_generated_images(generated_images)

    def __init__(self, images=None, batch_size=35, dropout=0.46, relu_alpha=0.2, momentum=0.9, random_values=100):
        self.random_values = random_values
        self.generator_input_shape = (self.random_values,)
        self.batch_size = batch_size
        self.noise_shape = (self.batch_size, self.random_values)
        if images is not None:
            self.images = np.asarray(images)
            self.num_images = len(images)
            print("Loaded {} images into GAN.".format(len(self.images)))
            self.discriminator_input_shape = (self.images[0].shape)
        else:
            print("GAN initialized without loading images (-f), training unavailable.")
            self.discriminator_input_shape = (256,256,3) #required for model specification

        #tunable model parameters
        self.default_epochs = 200
        self.generator_opt = RMSprop(lr=0.00005, clipvalue=1.0, decay=2e-9)
        self.discriminator_opt = RMSprop(lr=0.000072, clipvalue=1.0, decay=4e-9)
        self.dropout = dropout
        self.relu_alpha = relu_alpha
        self.momentum = momentum

        #create models for convinience
        self.generator = self.generator_model()
        self.discriminator = self.discriminator_model()
        self.adversarial = self.adversarial_model()

        #compile models
        self.discriminator.compile(loss='binary_crossentropy', optimizer=self.discriminator_opt)
        self.adversarial.compile(loss='binary_crossentropy', optimizer=self.generator_opt)
