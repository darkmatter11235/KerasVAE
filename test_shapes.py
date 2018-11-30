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

def is_cut_legal(cut, ncuts):
    mc_spacing = 0.2
    for ncut in ncuts:
        if abs(cut - ncut) >= mc_spacing or cut == ncut:
            continue
        else:
            return False
    return True

def get_neighboring_cuts(data, cut_map):

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
            cur_y = prev_y + step_size
            cur_x = prev_x
            next_track = current_track + 1
            if next_track in cut_map:
                cut_map[next_track].append(cur_x)
            else:
                cut_map[next_track] = [cur_x]
        elif ctype == 1:
            cur_y = prev_y + step_size
            cur_x = prev_x + mc_spacing
            next_track = current_track + 1
            if next_track in cut_map:
                cut_map[next_track].append(cur_x)
            else:
                cut_map[next_track] = [cur_x]
        elif ctype == 2:
            cur_y = prev_y + 2 * step_size
            cur_x = prev_x + mc_spacing
            next_track = current_track + 2
            if next_track in cut_map:
                cut_map[next_track].append(cur_x)
            else:
                cut_map[next_track] = [cur_x]


def generate_image_from_cuts(cut_map, output_file, distort=False):

    # Number of tracks
    n = 6
    # Create figure and axes
    fig, ax = plt.subplots(1)
    polygons = []
    y_step = 1.0 / n
    height = 1.0 / (2 * n)

    min_length = 0.4
    cut_width = 0.1

    # Loop over data points
    for i in range(n):

        y = y_step * i
        x = 0
        cut_locs = []
        if i in cut_map:
            cut_locs = cut_map[i]

        delta_width = 0
        delta_loc = 0
        for cut_loc in cut_locs:
            if distort:
                #delta_width = np.random.randint(-5, 5)/100
                delta_loc = np.random.randint(0, 10)/100
                cut_loc += delta_loc

            if cut_loc - x > min_length or ( cut_loc == cut_locs[0] and cut_loc < cut_width ):
                r = Rectangle((x, y), cut_loc - x, height)
                x = cut_loc + cut_width + delta_width
                polygons.append(r)
        if 1 - x > 0:
            r = Rectangle((x, y), 1 - x, height)
            polygons.append(r)

    pc = PatchCollection(polygons)
    pc.set_color("black")
    ax.add_collection(pc)
    plt.axis('off')
    # plt.show()
    plt.savefig(output_file)


def generate_seed_cut_map():
    # Number of tracks
    orient = "HORIZONTAL"
    n = 6
    # Create figure and axes
    y_step = 1.0 / n
    height = 1.0 / (2 * n)

    min_length = 0.4
    mc_spacing = 0.2
    cut_width = 0.1
    max_cuts = 3
    n_cut_types = 3
    data = {}
    data['orient'] = orient
    data['max_tracks'] = n
    data['mc_spacing'] = mc_spacing
    data['step_size'] = y_step
    cut_map = {}
    skip_neighbors = False

    # Loop over data points
    for i in range(n):

        y = y_step * i
        x = 0
        ncuts = np.random.randint(0, max_cuts, 1)[0]
        cut_type = np.random.randint(0, n_cut_types, 1)[0]
        cut_locs = []
        if i in cut_map:
            cut_locs = cut_map[i]
            skip_neighbors = True
        else:
            for j in range(ncuts):
                xloc = round(np.random.sample(1)[0], 2)
                if i - 1 in cut_map:
                    if is_cut_legal(xloc, cut_map[i - 1]):
                        cut_locs.append(xloc)
                else:
                    cut_locs.append(xloc)
                skip_neighbors = False
        cut_locs.sort()
        if len(cut_locs):
            cut_map[i] = cut_locs
        for cut_loc in cut_locs:
            if cut_loc - x > min_length or cut_loc < cut_width:
                width = cut_loc - x
                data['coords'] = [cut_loc, y, width, height]
                data['track_number'] = i
                data['cut_type'] = cut_type
                x = cut_loc + cut_width
                if not skip_neighbors:
                    get_neighboring_cuts(data, cut_map)

    return cut_map


seed_cuts = generate_seed_cut_map()

generate_image_from_cuts(seed_cuts, "test_cuts.png")
generate_image_from_cuts(seed_cuts, "test_cuts_distorted.png", True)
