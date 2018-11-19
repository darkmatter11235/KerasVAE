from PIL import Image
import numpy as np
import os
from keras.preprocessing.image import img_to_array, load_img


def load_image(infilename):
    img = Image.open(infilename)
    img.load()
    data = np.asarray(img, dtype="int32")
    return data


def save_image(npdata, outfilename):
    img = Image.fromarray(np.asarray(np.clip(npdata, 0, 255), dtype="uint8"), "L")
    img.save(outfilename)


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
        img.thumbnail((image_width, image_height))
        #img.show()
        #img.resize((image_width, image_height))
        # Convert to Numpy Array
        x = img_to_array(img)
        #print(x.shape)
        #x = x.reshape((num_channels, image_height, image_width))
        # Normalize
        # x = (x - 128.0) / 128.0
        dataset[i] = x
        i += 1
        if i % 250 == 0:
            print("%d images to array" % i)
    return dataset

