from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model, load_model
from utils import *

model = load_model("convAE_1000.h5")

# get the symbolic outputs of each key layer
layer_dict = dict([(layer.name, layer) for layer in model.layers])

print(layer_dict)

from keras import backend as K

layer_name = 'conv2d_4'

filter_index = 0

layer_output = layer_dict[layer_name].output

loss = K.mean(layer_output[:, :, :, filter_index])



