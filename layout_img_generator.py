import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

import os
import shutil


def generate_training_data(outFilePath):
    # Number of tracks
    n = 6
    add_cuts = False
    # Create figure and axes
    fig, ax = plt.subplots(1)
    polygons = []
    cuts = []
    y_step = 1.0 / n
    height = 1.0 / (2 * n)
    width = 1.0

    min_length = 0.2
    cut_width = 0.1
    cut_height = height

    max_cuts = 3
    # Loop over data points
    for i in range(n):
        y = y_step * i
        x = 0
        # rect = Rectangle((x, y), width, height, )
        # polygons.append(rect)
        ncuts = np.random.randint(0, max_cuts, 1)[0]
        # r1 = Rectangle((x, y), xloc, height)
        # r2 = Rectangle((x + cut_width, y), cut_width, cut_height)
        cut_locs = []
        # print(ncuts)
        for j in range(ncuts):
            xloc = round(np.random.sample(1)[0], 2)
            cut_locs.append(xloc)
        cut_locs.sort()

        for cut_loc in cut_locs:
            if cut_loc - x > min_length or cut_loc < cut_width:
                r = Rectangle((x, y), cut_loc - x, height)
                rc = Rectangle((cut_loc, y), cut_width, height)
                x = cut_loc + cut_width
                polygons.append(r)
                cuts.append(rc)
        if 1 - x > min_length:
            r = Rectangle((x, y), 1 - x, height)
            polygons.append(r)
        else:
            r = Rectangle((x, y), 1 - x, height)
            cuts.append(r)

    pc = PatchCollection(polygons)
    pc.set_color("black")
    ax.add_collection(pc)
    if add_cuts:
        pc = PatchCollection(cuts)
        pc.set_color("red")
        ax.add_collection(pc)
    plt.axis('off')
    # plt.show()
    plt.savefig(outFilePath)


def generate_random_training_image(outFilePath):
    # Number of tracks
    n = 6
    add_cuts = False
    # Create figure and axes
    fig, ax = plt.subplots(1)
    polygons = []
    cuts = []
    y_step = 1.0 / n
    height = 1.0 / (2 * n)
    width = 1.0

    min_length = 0.2
    cut_width = 0.1
    cut_height = height

    max_cuts = 3
    # Loop over data points
    for i in range(n):
        y = y_step * i
        x = 0
        # rect = Rectangle((x, y), width, height, )
        # polygons.append(rect)
        ncuts = np.random.randint(0, max_cuts, 1)[0]
        # r1 = Rectangle((x, y), xloc, height)
        # r2 = Rectangle((x + cut_width, y), cut_width, cut_height)
        cut_locs = []
        # print(ncuts)
        for j in range(ncuts):
            xloc = round(np.random.sample(1)[0], 2)
            cut_locs.append(xloc)
        cut_locs.sort()
        for cut_loc in cut_locs:
            if cut_loc - x > min_length or cut_loc < cut_width:
                r = Rectangle((x, y), cut_loc - x, height)
                rc = Rectangle((cut_loc, y), cut_width, height)
                x = cut_loc + cut_width
                polygons.append(r)
                cuts.append(rc)
        if 1 - x > min_length:
            r = Rectangle((x, y), 1 - x, height)
            polygons.append(r)
        else:
            r = Rectangle((x, y), 1 - x, height)
            cuts.append(r)

    pc = PatchCollection(polygons)
    pc.set_color("black")
    ax.add_collection(pc)
    if add_cuts:
        pc = PatchCollection(cuts)
        pc.set_color("red")
        ax.add_collection(pc)
    plt.axis('off')
    # plt.show()
    plt.savefig(outFilePath)
    # f = plt.figure()


train_img_dir = "./data/train"
test_img_dir = "./data/test"
validate_img_dir = "./data/scratch"
# train_img_dir = "./data1/train"
# test_img_dir = "./data1/test"
# validate_img_dir = "./data1/scratch"
n_train = 900
n_test = 100
n_validate = 1
os.makedirs(train_img_dir, exist_ok=True)
os.makedirs(test_img_dir, exist_ok=True)
os.makedirs(validate_img_dir, exist_ok=True)

for i in range(n_train):
    img_path = train_img_dir + "/train_" + str(i) + ".png"
    generate_random_training_image(img_path)

for i in range(n_test):
    img_path = test_img_dir + "/test_" + str(i) + ".png"
    generate_random_training_image(img_path)

"""
for i in range(n_validate):
    img_path = validate_img_dir+"/validate_"+str(i)+".png"
    generate_random_training_image(img_path)
"""
