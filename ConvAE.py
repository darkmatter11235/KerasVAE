from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model

# input layer will be image of shape (28,28,1)

input_img = Input(shape=(28, 28, 1,))

x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)  # (output_shape=(24,24,16)

x = MaxPooling2D((2, 2), padding='same')(x)  # (output_shape=(20,20,16)

x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)  # (output_shape=(16,16,8))

x = MaxPooling2D((2, 2), padding='same')(x)  # (output_shape=(12,12,8))

x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)  # (output_shape=(8,8,8))

encoded = MaxPooling2D((2, 2), padding='same')(x)  # (output_shape=(4,4,8))

x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)  # (output_shape=(24,24,16)

x = UpSampling2D((2, 2), padding='same')(x)  # (output_shape=(20,20,16)

x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)  # (output_shape=(16,16,8))

x = UpSampling2D((2, 2), padding='same')(x)  # (output_shape=(12,12,8))

x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)  # (output_shape=(8,8,8))

x = UpSampling2D((2, 2), padding='same')(x)  # (output_shape=(4,4,8))

decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)

# Load the mnist data set
(x_train, _), (x_test, _) = mnist.load_data()
