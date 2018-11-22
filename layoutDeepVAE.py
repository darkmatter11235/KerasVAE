from keras.layers import Input, Dense, Concatenate
from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img
import os
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

from utils import *


# encoding dimension size
encoding_dim_1 = 800

encoding_dim_2 = 400

encoding_dim_3 = 200

encoding_dim_4 = 100

# image dimensions
image_width = 160

image_height = 120

num_channels = 1

num_epochs = 5000

load_existing = False

ref_model_param = num_epochs

autoencoder = []
# input layer will be image of shape (28,28,1)
# Load the mnist data set
# (x_train, _), (x_test, _) = mnist.load_data()
if not load_existing:
    folder = "./data/train"
    x_train = img_files_to_np_array(folder, image_width, image_height, num_channels)
    x_train = x_train.astype('float32') / 255.
    x_train = x_train.reshape(len(x_train), np.prod(x_train.shape[1:]))
#   x_train = x_train.reshape(len(x_train), image_width, image_height, num_channels)

folder = "./data/test"
x_test = img_files_to_np_array(folder, image_width, image_height, num_channels)
x_test = x_test.astype('float32') / 255.
x_test = x_test.reshape(len(x_test), np.prod(x_test.shape[1:]))
#x_test = x_test.reshape(len(x_test), image_width, image_height, num_channels)

# input image size W*H
input_size = image_width*image_height*num_channels

# this is input placeholder
input_img = Input(shape=(input_size,))

x = Dense(encoding_dim_1, activation='relu')(input_img)

x = Dense(encoding_dim_2, activation='relu')(x)

x = Dense(encoding_dim_3, activation='relu')(x)

encoded = Dense(encoding_dim_4, activation='relu')(x)

x = Dense(encoding_dim_4, activation='relu')(encoded)

x = Dense(encoding_dim_3, activation='relu')(x)

x = Dense(encoding_dim_2, activation='relu')(x)

x = Dense(encoding_dim_1, activation='relu')(x)

decoded = Dense(input_size, activation='sigmoid')(x)

autoencoder = Model(input_img, decoded)

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

autoencoder.summary()

if not load_existing:
    # from keras.callbacks import TensorBoard
    autoencoder.fit(x_train, x_train,
                    epochs=num_epochs,
                    batch_size=256,
                    shuffle=True,
                    validation_data=(x_test, x_test))

# encoded_images = encoder.predict(x_test)
# print(encoded_images.shape)
# decoded_images = decoder.predict(encoded_images)
decoded_images = autoencoder.predict(x_test)

if load_existing:
    del autoencoder
    import h5py
    model_file = 'DeepAE_' + str(ref_model_param) + '.h5'
    # f = h5py.File(model_file, 'r')
    # print(f.attrs.get('keras_version'))
    autoencoder = load_model(model_file)

decoded_images = autoencoder.predict(x_test)

if not load_existing:
    autoencoder.save("./convAE_" + str(num_epochs) + ".h5")

n = 1
snap_image_to_bw = False
snap_image_to_bw = True

plt.figure(figsize=(20, 10))
for i in range(n):
    # display original image
    ax = plt.subplot(2, n, i + 1)
    if num_channels > 1:
        plt.imshow(x_test[i].reshape(image_height, image_width, num_channels))
    else:
        plt.gray()
        plt.imshow(x_test[i].reshape(image_height, image_width))
    # plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    if num_channels > 1:
        plt.imshow(decoded_images[i].reshape(image_height, image_width, num_channels))
    else:
        plt.gray()
        img = decoded_images[i]
        img = img * 255
        if snap_image_to_bw:
            img[img < 128] = 0
            img[img >= 128] = 255
        plt.imshow(img.reshape(image_height, image_width))
    # plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()
