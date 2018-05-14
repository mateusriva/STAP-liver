"""This script assembles and compares SRGs, but splits the background into
three distinct labels, and groups the liver into a single one.
"""

import numpy as np
from time import time
from copy import deepcopy
import matplotlib.pyplot as plt, matplotlib.colors as mcolors, matplotlib.patches as mpatches
from skimage.morphology import ball
import scipy.ndimage as ndi
from itertools import permutations

from lic_srg import SRG, Matching
from lic_patient import Patient, Volume, LabelMap
from lic_display import display_volume, display_volumes, IndexTracker

label_color_map = {
    0: (0,0,0),         # Background posterior: no label
    1: (0.5,0.5,0.5),   # Background anterior: gray
    2: (1,1,1),         # Background body: white
    3: (0,0,1),         # Vena Cava: blue
    4: (0,1,1),         # Portal Vein: light blue
    5: (0.5,0,1),       # Left Hepatic Vein: purplish-blue
    6: (0,0.5,1),       # Middle Hepatic Vein: light-but-not-so-much blue
    7: (0,0,0.5),       # Right Hepatic Vein: dark blue
    8: (1,0,0)         # Liver: red
}

# Importing patient
model_patient = Patient.build_from_folder("data/4")

# Reassembling labelmap as: 0: bgNorth, 1: bgSouth, 2: bgBody, 3-7: veins, 8 liver
labelmap = model_patient.labelmaps["t2"]
labelmap.data += 2 # veins are 3-8, not 1-5
labelmap.data[labelmap.data >= 8] = 8 # all liver segments are label 8
labelmap.data[np.logical_and(model_patient.volumes["t2"].data < 10, labelmap.data == 2)] = 0 # black background is 0
labelmap.data[labelmap.data.shape[1]//2:,:,:][labelmap.data[labelmap.data.shape[1]//2:,:,:] == 0] = 1 # anterior background is 1
model_patient.labelmaps["t2"] = labelmap

#display_volume(labelmap.data, title="Simplified Labelmap", cmap=mcolors.ListedColormap(list(label_color_map.values())))

print("Building model graph... ", end="", flush=True)
t0 = time()
model_graph = SRG.build_from_patient(model_patient)
print("Done. {:.4f}s".format(time()-t0))

print("Running watershed... ", end="", flush=True)
t0 = time()
watershed_labelmap = model_patient.volumes['t2'].watershed_volume()
print("Done. {} labels found. {:.4f}s".format(watershed_labelmap.header["num_labels"], time()-t0))

print("Building observation graph... ", end="", flush=True)
t0 = time()
observed_patient = deepcopy(model_patient)
observed_patient.labelmaps['t2'] = watershed_labelmap
observation_graph = SRG.build_from_patient(observed_patient)
print("Done. {:.4f}s".format(time()-t0))

# Testing multiple vertex cost weights
for weights in [(0,1),(0.2,0.8),(0.4,0.6),(1,1),(0.6,0.4),(0.8,0.2),(1,0)]:
    # generating greedy solution
    # TODO: move this to a "lic_solution" module?
    print("Generating greedy solution... ", end="", flush=True)
    t0 = time()
    # creating empty match dict
    match_dict = {}
    # for each vertex in the observation graph, find the closest matched model vertex (ignore edge info)
    for i, obs_vertex in enumerate(observation_graph.vertexes):
        best_model_vertex = np.argmin([obs_vertex.cost_to(model_vertex, weights=weights) for model_vertex in model_graph.vertexes])
        match_dict[i] = best_model_vertex
    print("Done. {:.4f}s".format(time()-t0))

    print("Computing cost... ", end="", flush=True)
    solution = Matching(match_dict, model_graph, observation_graph)
    cost = solution.cost()
    print("Done. Cost is {}. {:.4f}s".format(cost, time()-t0))

    # Assembling and displaying predicted labelmap
    predicted_labelmap = deepcopy(watershed_labelmap.data)
    for observation, prediction in match_dict.items():
        predicted_labelmap[predicted_labelmap==observation] = -prediction
    predicted_labelmap *= -1

    fig,axes=plt.subplots(1,2)
    trackers = [IndexTracker(ax, X, title=title, cmap=cmap) 
                for ax,X,title,cmap in zip(axes,[predicted_labelmap, model_patient.labelmaps["t2"].data],["Predictions {}".format(weights),"Truth"],[mcolors.ListedColormap(list(label_color_map.values())),mcolors.ListedColormap(list(label_color_map.values()))])]
    for tracker in trackers:
        fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    plt.show()