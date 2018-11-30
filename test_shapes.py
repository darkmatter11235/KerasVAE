from shapely.geometry import Polygon, Point
from descartes import PolygonPatch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

import os
import shutil


# type-0 - aligned & adj
# type-1 - MS & adj
# type-2 - MS & ISO
# algorithm pick the first cut and randomly choose the type for the next cut
# if you get type 3, the cut will skip a track

def get_neighboring_cuts(data, got_cuts):

    coords = data['coords']
    step_size = data['step_size']
    orient = data['orient']
    current_track = data['track_number']
    n = data['max_tracks']
    mc_spacing = data['mc_spacing']
    ctype = data['cut_type']

    # boundary conditions
    if (ctype == 0 or ctype == 1) and (current_track == n):
        pass

    if ctype == 2 and current_track >= (n - 1):
        pass

    if orient == "HORIZONTAL":

        prev_x = coords[0]
        prev_y = coords[1]
        prev_width = coords[2]
        prev_height = coords[3]
        cur_width = prev_width
        cur_height = prev_height

        if ctype == 0:
            cur_y = prev_y * step_size
            cur_x = prev_x
            next_track = current_track + 1
            if next_track in got_cuts:
                got_cuts[next_track].append(cur_x)
            else:
                got_cuts[next_track] = [cur_x]
        elif ctype == 1:
            cur_y = prev_y * step_size
            cur_x = prev_x + mc_spacing
            next_track = current_track + 1
            if next_track in got_cuts:
                got_cuts[next_track].append(cur_x)
            else:
                got_cuts[next_track] = [cur_x]
        elif ctype == 2:
            cur_y = prev_y * 2 * step_size
            cur_x = prev_x + mc_spacing
            next_track = current_track + 2
            if next_track in got_cuts:
                got_cuts[next_track].append(cur_x)
            else:
                got_cuts[next_track] = [cur_x]



def generate_training_data(outFilePath):
    # Number of tracks
    orient = "HORIZONTAL"
    n = 6
    add_cuts = False
    # Create figure and axes
    fig, ax = plt.subplots(1)
    polygons = []
    cuts = []
    y_step = 1.0 / n
    height = 1.0 / (2 * n)
    width = 1.0

    min_length = 0.4
    mc_spacing = 0.2
    cut_width = 0.1
    cut_height = height
    max_cuts = 3
    n_cut_types = 3
    data = {}
    data['orient'] = orient
    data['max_tracks'] = n
    data['mc_spacing'] = mc_spacing
    data['step_size'] = y_step
    got_cuts = {}
    skip_neighbors = False

    # Loop over data points
    for i in range(n):

        y = y_step * i
        x = 0
        ncuts = np.random.randint(0, max_cuts, 1)[0]
        cut_type = np.random.randint(0, n_cut_types, 1)[0]
        cut_locs = []
        if i in got_cuts:
            cut_locs = got_cuts[i]
            skip_neighbors = True
        else:
            for j in range(ncuts):
                xloc = round(np.random.sample(1)[0], 2)
                cut_locs.append(xloc)
        cut_locs.sort()
        print(cut_locs)
        for cut_loc in cut_locs:
            if cut_loc - x > min_length or cut_loc < cut_width:
                width = cut_loc - x
                xl = x - width / 2
                xh = x + width / 2
                yl = y - height / 2
                yh = y + height / 2
                data['coords'] = [x, y, width, height]
                data['track_number'] = i
                data['cut_type'] = cut_type
                r = Rectangle((x, y), cut_loc - x, height)
                rc = Rectangle((cut_loc, y), cut_width, height)
                x = cut_loc + cut_width
                #shape_rect = Polygon([(xl, yl), (xl, yh), (xh, yh), (xh, yl)])
                polygons.append(r)
                cuts.append(rc)
                if not skip_neighbors:
                    get_neighboring_cuts(data, got_cuts)
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


generate_training_data("test_cuts.png")
