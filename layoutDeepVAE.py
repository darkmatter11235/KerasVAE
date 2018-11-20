from keras.layers import Input, Dense, Concatenate
from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img
import os
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

from utils import *


# encoding dimension size
encoding_dim = 32

encoding_dim_1 = 32

# image dimensions
image_width = 80
image_height = 60
num_channels = 3

# input image size W*H
input_size = image_width*image_height*num_channels

# this is input placeholder
input_img = Input(shape=(input_size,))

x = Dense(encoding_dim_1, activation='relu')(input_img)

encoded = Dense(encoding_dim, activation='relu')(x)

x = Dense(encoding_dim_1, activation='sigmoid')(encoded)

decoded = Dense(input_size, activation='sigmoid')(x)

autoencoder = Model(input_img, decoded)

encoder = Model(input_img, encoded)

# create a placeholder for encoded input
encoded_input = Input(shape=(encoding_dim,))

decoder_layer_first= autoencoder.layers[-2]

decoder_layer_second = autoencoder.layers[-1]

decoder = Model(encoded_input, decoder_layer_second(decoder_layer_first(encoded_input)))

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')



folder = "./data/train"
x_train = img_files_to_np_array(folder, image_width, image_height, num_channels)

folder = "./data/test"
x_test = img_files_to_np_array(folder, image_width, image_height, num_channels)

#(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.

x_test = x_test.astype('float32') / 255.

x_train = x_train.reshape(len(x_train), np.prod(x_train.shape[1:]))

x_test = x_test.reshape(len(x_test), np.prod(x_test.shape[1:]))

# print(x_train.shape)

autoencoder.fit(x_train, x_train,
                epochs=100,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))

encoded_images = encoder.predict(x_test)
decoded_images = decoder.predict(encoded_images)


n = 10

plt.figure(figsize=(20, 10))
for i in range(n):
    # display original image
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(image_height, image_width, num_channels))
    #plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_images[i].reshape(image_height, image_width, num_channels))
    #plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()
