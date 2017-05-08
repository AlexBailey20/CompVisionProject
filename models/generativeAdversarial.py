# Shishir Tandale

import numpy as np
import math
import matplotlib.pyplot as plt

from PIL import Image
from keras import backend as tf
from keras.models import Sequential
from keras.constraints import maxnorm
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
            plt.savefig("images/gan/{}_{:03d}.png".format(title, num))
    else:
        plt.imshow(gen_images[0])
        plt.savefig("images/gan/{}.png".format(title))

class model_guard(object):
    #protects a model's parameters from changing during training
    #use in a with statement, passing the model you want to protect temporarily
    #used in GAN
    def __init__(self, model):
        self.model = model
    def __enter__(self):
        self.model.trainable = False
        for layer in self.model.layers:
            layer.trainable = False
    def __exit__(self, exc_type, exc_value, traceback):
        self.model.trainable = True
        for layer in self.model.layers:
            layer.trainable = True
class GAN_Model(object):
    #inspired mostly by https:/medium.com/towards-data-science/gan-by-example-using-keras-on-tensorflow-backend-1a6d515a60d0
    def add_residual_block_h(self, model, cnn_scale, radius=6, relu=True, scale=False, input_shape=None):
        if input_shape is not None:
            model.add(Conv2DTranspose(cnn_scale, radius, padding='same', input_shape=input_shape))
        else:
            model.add(Conv2DTranspose(cnn_scale, radius, padding='same'))
        model.add(BatchNormalization(momentum=self.momentum))
        if scale:
            model.add(UpSampling2D(2))
        if relu:
            model.add(Activation('relu'))
    def add_residual_block(self, model, conv_scale, final_dim=None):
        self.add_residual_block_h(model, int(self.filter_dim/4*conv_scale), radius=1)
        self.add_residual_block_h(model, int(self.filter_dim/4*conv_scale), radius=3)
        if final_dim is None:
            self.add_residual_block_h(model, self.filter_dim*conv_scale, radius=1, scale=True)
        else:
            self.add_residual_block_h(model, final_dim, relu=False, radius=1, scale=True)
    def generator_model(self, num_layers=3):
        #Input: 64 floats
        #Output: 256x256x3 floats
        model = Sequential()
        model.add(Reshape(self.random_shape, input_shape=self.generator_input_shape))
        model.add(Dropout(self.dropout))
        model.add(BatchNormalization(momentum=self.momentum))
        for i, scale in [(i, 2**x) for (i, x) in enumerate(range(num_layers))]:
            self.add_residual_block(model, scale, final_dim=3 if i==num_layers-1 else None)
        model.add(Activation('sigmoid'))
        return model
    def discriminator_model(self, num_layers=3):
        #Input: 256x256x3 floats
        #Output: {0,1}
        model = Sequential()
        for i, scale in [(i, 2**x) for (i, x) in enumerate(range(num_layers)[-1::-1])]:
            if i==0:
                model.add(Conv2D(self.filter_dim*scale, 7, padding='same', input_shape=self.discriminator_input_shape))
            else:
                model.add(Conv2D(self.filter_dim*scale, 8, strides=4, padding='same'))
            model.add(LeakyReLU(self.relu_alpha))
            model.add(Dropout(self.dropout))
        model.add(Flatten())
        model.add(Dense(1))
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
            self.generator.load_weights(self.gen_savefilename)
            self.discriminator.load_weights(self.dis_savefilename)
            print("Loaded weights, resuming training.")

        #pre-train discriminator
        random_range = lambda start=0, end=self.num_images, count=self.batch_size: np.random.randint(start, end, size=count)
        image_batch = self.images[random_range()]
        generated_images = self.generator.predict_on_batch(self.random_noise())
        X = np.concatenate((image_batch, generated_images))
        ones = np.ones([self.batch_size])
        ones_zeros = np.append(ones, (np.zeros([self.batch_size])))
        _ = self.discriminator.fit(X, ones_zeros, batch_size=self.batch_size, verbose=0)

        #train for multiple epochs
        def train(epochs):
            generated_images = None #reference for saving results of each epoch
            try:
                for e in range(1, epochs+1):
                    header_text = "{}\t\t\t{}".format("Discriminator","Generator") if e==1 else ""
                    print("{}\t{}\t\t{}".format("Epoch", e, header_text))
                    for b in range(self.num_batches):
                        #in wasserstein you train discriminator n times for each generator update
                        for _ in range(5):
                            #collect real and generated images
                            image_batch = self.images[random_range()]
                            generated_images = self.generator.predict_on_batch(self.random_noise())
                            X = np.concatenate((image_batch, generated_images))
                            #train discriminator to recognize which images are generated
                            d_loss = self.discriminator.fit(X, ones_zeros, batch_size=self.batch_size, verbose=0)
                        #train generator to fool discriminator on generated images
                        #but be careful not to modify discriminator parameters
                        with model_guard(self.discriminator):
                            g_loss = self.adversarial.train_on_batch(self.random_noise(), ones)
                        #print losses and save weights/images
                        if b % 1 == 0:
                            batch_text = "Batch" if b==0 else ""
                            print("{}\t{}\t\t{}\t\t{}".format(batch_text, b, d_loss.history['loss'][-1], g_loss))
                        if b % 5 == 0:
                            save_generated_images(generated_images, title="epoch_{:02d}_batch_{:02d}".format(e, b), limit=1)
            except KeyboardInterrupt:
                print("\nUser stopped training.")
            finally:
                self.generator.save_weights(self.gen_savefilename)
                self.discriminator.save_weights(self.dis_savefilename)
        train(epochs)

    def generate(self, num_samples=None, with_discriminator=False):
        if num_samples is None:
            num_samples = self.batch_size
        #make some noise
        noise = self.random_noise(num=num_samples)
        #load generator weights
        self.generator.load_weights(self.gen_savefilename)
        if with_discriminator:
            #run this batch through the adversarial and update weights
            self.discriminator.load_weights(self.dis_savefilename)
            g_loss = self.adversarial.train_on_batch(noise, np.ones([num_samples]))
            print("Discriminator yielded loss {}".format(g_loss))
        #generate and save images
        generated_images = self.generator.predict_on_batch(noise)
        save_generated_images(generated_images)
        print("Generated {} samples.".format(num_samples))

    def __init__(self, images=None, batch_size=30, dropout=0.33, relu_alpha=0.19, momentum=0.9, print_summary=False, scale=256, layers=4):
        base_dim = int(scale/2**layers)
        print(f"Using {base_dim} random values for generation.")
        self.random_shape = (base_dim, base_dim, 1)
        self.random_values = base_dim**2
        self.generator_input_shape = (self.random_values,)
        self.batch_size = batch_size
        self.noise_shape = (self.batch_size, self.random_values)
        if images is not None:
            self.images = np.asarray(images)
            self.num_images = len(images)
            self.num_batches = int(self.num_images/self.batch_size)
            print("Loaded {} images into GAN.".format(len(self.images)))
            self.discriminator_input_shape = (self.images[0].shape)
        else:
            print("GAN initialized without loading images (-f), training unavailable.")
            self.discriminator_input_shape = (scale,scale,3) #required for model specification

        #tunable model parameters
        self.filter_dim = 10
        self.num_layers = layers
        self.gen_savefilename = 'generator.h5'
        self.dis_savefilename = 'discriminator.h5'
        self.dropout = dropout
        self.relu_alpha = relu_alpha
        self.momentum = momentum

        #create models for convinience
        self.generator = self.generator_model(num_layers=self.num_layers)
        self.discriminator = self.discriminator_model(num_layers=self.num_layers)
        if print_summary:
            self.generator.summary()
            self.discriminator.summary()
        self.adversarial = self.adversarial_model()

        #compile models
        #wasserstein_discriminator_loss = lambda true, pred: tf.abs(tf.mean(pred)-tf.mean(true))
        wasserstein_adversary_loss = lambda true, pred: tf.abs(tf.mean(pred))
        content_loss = 'mean_squared_error'
        adversarial_loss = wasserstein_adversary_loss
        self.discriminator.compile(loss=content_loss, optimizer=RMSprop(lr=0.00004))
        self.adversarial.compile(loss=adversarial_loss, optimizer=RMSprop(lr=0.00003, clipvalue=0.017))
