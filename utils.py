from PIL import Image
import numpy as np
import os
from keras.preprocessing.image import img_to_array, load_img
import keras.backend as K
from keras.objectives import binary_crossentropy


def load_image(infilename):
    img = Image.open(infilename)
    img.load()
    data = np.asarray(img, dtype="int32")
    return data


def save_image(npdata, outfilename):
    img = Image.fromarray(np.asarray(np.clip(npdata, 0, 255), dtype="uint8"), "L")
    img.save(outfilename)


def expandedSobel(inputTensor):
    # this contains both X and Y sobel filters in the format (3,3,1,2)
    # size is 3 x 3, it considers 1 input channel and has two output channels: X and Y
    sobelFilter = K.variable([[[[1., 1.]], [[0., 2.]], [[-1., 1.]]],
                              [[[2., 0.]], [[0., 0.]], [[-2., 0.]]],
                              [[[1., -1.]], [[0., -2.]], [[-1., -1.]]]])
    # this considers data_format = 'channels_last'
    inputChannels = K.reshape(K.ones_like(inputTensor[0, 0, 0, :]), (1, 1, -1, 1))
    # if you're using 'channels_first', use inputTensor[0,:,0,0] above
    return sobelFilter * inputChannels


def sobelLoss(yTrue, yPred):
    # get the sobel filter repeated for each input channel
    filt = expandedSobel(yTrue)

    # calculate the sobel filters for yTrue and yPred
    # this generates twice the number of input channels
    # a X and Y channel for each input channel
    sobelTrue = K.depthwise_conv2d(yTrue, filt)
    sobelPred = K.depthwise_conv2d(yPred, filt)

    # now you just apply the mse:
    return K.mean(K.square(sobelTrue - sobelPred))


def xent_sobel(yTrue, yPred):
    xent_loss = binary_crossentropy(yTrue, yPred)
    sobel_loss = sobelLoss(yTrue, yPred)
    return xent_loss + sobel_loss


def img_files_to_np_array(folder, image_width, image_height, num_channels):
    files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    print("# of training images " + str(len(files)))
    # dataset = np.ndarray(shape=(len(files), num_channels, image_height, image_width),
    #                     dtype=np.float32)

    dataset = np.ndarray(shape=(len(files), image_height, image_width, num_channels),
                         dtype=np.uint8)
    i = 0
    for _file in files:
        img = load_img(folder + "/" + _file)  # this is a PIL image
        if num_channels == 1:
            img = img.convert("L")
        img.thumbnail((image_width, image_height))
        # img.show()
        # img.resize((image_width, image_height))
        # Convert to Numpy Array
        x = img_to_array(img)
        # print(x.shape)
        # x = x.reshape((num_channels, image_height, image_width))
        # Normalize
        # x = (x - 128.0) / 128.0
        x[x < 128] = 0
        x[x >= 128] = 255
        dataset[i] = x
        i += 1
        if i % 250 == 0:
            print("%d images to array" % i)
    return dataset


"""
folder = "./data/train"
_file = "train_10.png"
img = load_img(folder + "/" + _file)  # this is a PIL image
img = img.convert("L")
a = img_to_array(img)
a[a < 128] = 0
a[a >= 128] = 255
m = a < 255
# print(a[m[:,250,:]])
print(a[:, 240, :][m[:, 240, :]])
"""
