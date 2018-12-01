from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.datasets import mnist
import numpy as np
import keras.backend as K
import matplotlib.pyplot as plt
from keras.models import Model, load_model
from utils import *

# image_width = 28
image_width = 160

# image_height = 28
image_height = 120

num_channels = 1
# num_channels = 3

num_epochs = 5000

load_existing = False
#load_existing = True

ref_model_param = num_epochs

nfilters_L1 = 64

nfilters_L2 = 32

nfilters_L3 = 32

n_distortions = 4

# input layer will be image of shape (28,28,1)
# Load the mnist data set
# (x_train, _), (x_test, _) = mnist.load_data()
if not load_existing:
    folder = "./data/train/source"
    x_train_source = img_files_to_np_array(folder, image_width, image_height, num_channels)
    x_train_source = x_train_source.astype('float32') / 255.
    x_train_source = x_train_source.reshape(len(x_train_source), image_width, image_height, num_channels)
    folder = "./data/train/target"
    x_train_t = img_files_to_np_array(folder, image_width, image_height, num_channels)
    x_train_t = x_train_t.astype('float32') / 255.
    x_train_t = x_train_t.reshape(len(x_train_t), image_width, image_height, num_channels)
    x_train_target = np.ones_like(x_train_source)
    for i in range(len(x_train_t)):
        for k in range(n_distortions):
            x_train_target[n_distortions * i + k] = x_train_t[i]

folder = "./data/test/source"
x_test_source = img_files_to_np_array(folder, image_width, image_height, num_channels)
x_test_source = x_test_source.astype('float32') / 255.
x_test_source = x_test_source.reshape(len(x_test_source), image_width, image_height, num_channels)
folder = "./data/test/target"
x_test_t = img_files_to_np_array(folder, image_width, image_height, num_channels)
x_test_t = x_test_t.astype('float32') / 255.
x_test_t = x_test_t.reshape(len(x_test_t), image_width, image_height, num_channels)
x_test_target = np.ones_like(x_test_source)

for i in range(len(x_test_t)):
    for k in range(n_distortions):
        x_test_target[n_distortions * i + k] = x_test_t[i]

# input_img = Input(shape=(28, 28, 1,))
input_img = Input(shape=(image_width, image_height, num_channels,))

x = Conv2D(nfilters_L1, (3, 3), activation='relu', padding='same')(input_img)  # (output_shape=(24,24,16)

x = MaxPooling2D((2, 2), padding='same')(x)  # (output_shape=(20,20,16)

x = Conv2D(nfilters_L2, (3, 3), activation='relu', padding='same')(x)  # (output_shape=(16,16,8))

x = MaxPooling2D((2, 2), padding='same')(x)  # (output_shape=(12,12,8))

x = Conv2D(nfilters_L3, (3, 3), activation='relu', padding='same')(x)  # (output_shape=(8,8,8))

encoded = MaxPooling2D((2, 2), padding='same')(x)  # (output_shape=(4,4,8))

x = Conv2D(nfilters_L3, (3, 3), activation='relu', padding='same')(encoded)  # (output_shape=(24,24,16)

x = UpSampling2D((2, 2))(x)  # (output_shape=(20,20,16)

x = Conv2D(nfilters_L2, (3, 3), activation='relu', padding='same')(x)  # (output_shape=(16,16,8))

x = UpSampling2D((2, 2))(x)  # (output_shape=(12,12,8))

x = Conv2D(nfilters_L1, (3, 3), activation='relu', padding='same')(x)  # (output_shape=(8,8,8))

x = UpSampling2D((2, 2))(x)  # (output_shape=(4,4,8))

# decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)


def chamfer_loss_value(y_true, y_pred):
    # flatten the batch
    y_true_f = K.batch_flatten(y_true)
    y_pred_f = K.batch_flatten(y_pred)

    y_pred_mask_f = K.sigmoid(y_pred_f - 0.5)

    finalChamferDistanceSum = K.sum(y_pred_mask_f * y_true_f, axis=1, keepdims=True)

    return K.mean(finalChamferDistanceSum)


autoencoder.compile(optimizer='adadelta', loss=xent_sobel)
# autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
# autoencoder.compile(optimizer='adadelta', loss=chamfer_loss_value)

encoder = Model(input_img, encoded)

# encoded_input = Input(shape=(4, 4, 8))
# encoded_input = Input(shape=(20, 15, 8))
#encoded_input = Input(shape=(20, 15, 16))
encoded_input = Input(shape=(20, 15, 32))

decoder_layer = autoencoder.layers[-7]

decoder = Model(encoded_input, decoder_layer(encoded_input))

if not load_existing:
    # from keras.callbacks import TensorBoard
    autoencoder.fit(x_train_source, x_train_target,
                    epochs=num_epochs,
                    batch_size=256,
                    shuffle=True,
                    validation_data=(x_test_source, x_test_target))

#                callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])


# encoded_images = encoder.predict(x_test)
# print(encoded_images.shape)
# decoded_images = decoder.predict(encoded_images)

if load_existing:
    del autoencoder
    import h5py

    model_file = 'convAE_' + str(ref_model_param) + '.h5'
    # f = h5py.File(model_file, 'r')
    # print(f.attrs.get('keras_version'))
    # autoencoder = load_model(model_file)
    autoencoder = load_model(model_file, custom_objects={'xent_sobel': xent_sobel})

decoded_images = autoencoder.predict(x_test_source)

if not load_existing:
    autoencoder.save("./convAE_" + str(num_epochs) + ".h5")

n = 5
snap_image_to_bw = False
snap_image_to_bw = True

plt.figure(figsize=(20, 10))
for i in range(n):
    # display original image
    ax = plt.subplot(2, n, i + 1)
    if num_channels > 1:
        plt.imshow(x_test_target[i].reshape(image_height, image_width, num_channels))
    else:
        plt.gray()
        plt.imshow(x_test_target[i].reshape(image_height, image_width))
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
