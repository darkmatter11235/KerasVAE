from keras.layers import Input, Dense
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

# encoding dimension size
encoding_dim = 32

# input image size 28x28
input_size = 784

# this is input placeholder
input_img = Input(shape=(input_size,))

encoded = Dense(encoding_dim, activation='relu')(input_img)

decoded = Dense(input_size, activation='sigmoid')(encoded)

autoencoder = Model(input_img, decoded)

encoder = Model(input_img, encoded)

# create a placeholder for encoded input
encoded_input = Input(shape=(encoding_dim,))

decoder_layer = autoencoder.layers[-1]

decoder = Model(encoded_input, decoder_layer(encoded_input))

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

train_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    'data/train',
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary')

test_datagen = ImageDataGenerator(rescale=1. / 255)

test_generator = train_datagen.flow_from_directory(
    'data/train',
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary')


#(x_train, _), (x_test, _) = mnist.load_data()

#x_train = x_train.astype('float32') / 255.

#x_test = x_test.astype('float32') / 255.

#x_train = x_train.reshape(len(x_train), np.prod(x_train.shape[1:]))

#x_test = x_test.reshape(len(x_test), np.prod(x_test.shape[1:]))

# print(x_train.shape)

# autoencoder.fit(x_train, x_train,
autoencoder.fit(train_generator, train_generator,
                epochs=50,
                batch_size=256,
                shuffle=True,
                validation_data=(test_generator, test_generator))
# validation_data=(x_test, x_test))

#encoded_images = encoder.predict(x_test)
#decoded_images = decoder.predict(encoded_images)


n = 10
"""

plt.figure(figsize=(20, 10))
for i in range(n):
    # display original image
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_images[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()
"""