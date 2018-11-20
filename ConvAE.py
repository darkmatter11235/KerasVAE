from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from utils import *

#image_width = 28
image_width = 160

#image_height = 28
image_height = 120

num_channels = 1
#num_channels = 3

num_epochs = 50

# input layer will be image of shape (28,28,1)

#input_img = Input(shape=(28, 28, 1,))
input_img = Input(shape=(image_width, image_height, num_channels,))

x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)  # (output_shape=(24,24,16)

x = MaxPooling2D((2, 2), padding='same')(x)  # (output_shape=(20,20,16)

x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)  # (output_shape=(16,16,8))

x = MaxPooling2D((2, 2), padding='same')(x)  # (output_shape=(12,12,8))

x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)  # (output_shape=(8,8,8))

encoded = MaxPooling2D((2, 2), padding='same')(x)  # (output_shape=(4,4,8))

x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)  # (output_shape=(24,24,16)

x = UpSampling2D((2, 2))(x)  # (output_shape=(20,20,16)

x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)  # (output_shape=(16,16,8))

x = UpSampling2D((2, 2))(x)  # (output_shape=(12,12,8))

x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)  # (output_shape=(8,8,8))

x = UpSampling2D((2, 2))(x)  # (output_shape=(4,4,8))

# decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

encoder = Model(input_img, encoded)

# encoded_input = Input(shape=(4, 4, 8))
encoded_input = Input(shape=(20, 15, 8))

decoder_layer = autoencoder.layers[-7]

decoder = Model(encoded_input, decoder_layer(encoded_input))

# Load the mnist data set

folder = "./data/train"
x_train = img_files_to_np_array(folder, image_width, image_height, num_channels)

folder = "./data/test"
x_test = img_files_to_np_array(folder, image_width, image_height, num_channels)

#(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.

x_test = x_test.astype('float32') / 255.

# x_train = x_train.reshape(len(x_train), np.prod(x_train.shape[1:]))

# x_test = x_test.reshape(len(x_test), np.prod(x_test.shape[1:]))
x_train = x_train.reshape(len(x_train), image_width, image_height, num_channels)

x_test = x_test.reshape(len(x_test), image_width, image_height, num_channels)
#(x_train, _), (x_test, _) = mnist.load_data()
#x_train = x_train.astype('float32') / 255.
#x_test = x_test.astype('float32') / 255.
#x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))  # adapt this if using `channels_first` image data format
#x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))  # adapt this if using `channels_first` image data format

# from keras.callbacks import TensorBoard

autoencoder.fit(x_train, x_train,
                epochs=num_epochs,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))

#                callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])


#encoded_images = encoder.predict(x_test)
#print(encoded_images.shape)
#decoded_images = decoder.predict(encoded_images)
decoded_images = autoencoder.predict(x_test)

n = 10

plt.figure(figsize=(20, 10))
for i in range(n):
    # display original image
    ax = plt.subplot(2, n, i + 1)
    if num_channels > 1 :
        plt.imshow(x_test[i].reshape(image_height, image_width, num_channels))
    else:
        plt.imshow(x_test[i].reshape(image_height, image_width))
    # plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    if num_channels > 1 :
        plt.imshow(decoded_images[i].reshape(image_height, image_width, num_channels))
    else:
        plt.imshow(decoded_images[i].reshape(image_height, image_width))
    # plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()
