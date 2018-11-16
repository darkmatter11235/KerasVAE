from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator

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

test_datagen = ImageDataGenerator(rescale=1. / 255)

test_generator = test_datagen.flow_from_directory(
    'data/train',
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary')

train_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    'data/train',
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    'data/validation',
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary')

# (x_train, _), (x_test, _) = mnist.load_data()
