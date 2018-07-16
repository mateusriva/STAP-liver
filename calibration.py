"""Functions for calibrating the pipeline with dummies.

This module contains both functions for generating dummies,
and for putting them through the segmentation pipeline."""

import numpy as np
import math
from scipy.ndimage.measurements import center_of_mass as measure_center_of_mass
from time import time
from copy import deepcopy
from itertools import product
from skimage.morphology import watershed, h_minima, ball, local_minima
from skimage.color import rgb2gray
import scipy.ndimage as ndi
from matplotlib.colors import ListedColormap
from skimage.util import random_noise

from srg import SRG
from display_utils import display_volume, display_segments_as_lines, display_solution, represent_srg

def generate_dummy(flavor):
    """Generates dummy data.

    Flavor generates specific dummies:
    * 0 : generates an 32x32x8 dummy, with 4 homogeneous classes in 8x8x8 towers.
    * 1 : similar to 0, but the first class is broken in two.
    * 2 : similar to 0, but infiltration occurs
    """
    if flavor == 0:
        dummy = np.empty((32,32,8))
        dummy[:16,:16,:] = 16
        dummy[:16,16:,:] = 32
        dummy[16:,:16,:] = 48
        dummy[16:,16:,:] = 64
        return dummy
    if flavor == 1:
        dummy = np.empty((32,32,8))
        dummy[:8,:16,:] = 12
        dummy[8:16,:16,:] = 20
        dummy[:16,16:,:] = 32
        dummy[16:,:16,:] = 48
        dummy[16:,16:,:] = 64
        return dummy
    if flavor == 2:
        dummy = np.empty((32,32,8))
        dummy[:16,:16,:] = 16
        dummy[:16,16:,:] = 32
        dummy[16:24,:16,:] = 48
        dummy[24:,:16,:] = 16
        dummy[16:,16:,:] = 64
        return dummy
    if flavor == 3:
        dummy = np.zeros((32,32,32))
        dummy[1:12,1:12,1:12] = ball(5)*100
        dummy[14:31,14:31,14:31] = ball(8)*200
        return dummy

def generate_dummy_label(flavor):
    """Generates dummy label data.

    Flavor generates specific dummies:
    * 0 : generates an 32x32x8 dummy, with 4 homogeneous classes in 16x16x8 towers.
    * 1 : similar to 0, but the first class is broken in two.
    """
    if flavor == 0:
        dummy = np.empty((32,32,8))
        dummy[:16,:16,:] = 0
        dummy[:16,16:,:] = 1
        dummy[16:,:16,:] = 2
        dummy[16:,16:,:] = 3
        return dummy
    if flavor == 1:
        dummy = np.empty((32,32,8))
        dummy[:8,:16,:] = 0
        dummy[8:16,:16,:] = 1
        dummy[:16,16:,:] = 2
        dummy[16:,:16,:] = 3
        dummy[16:,16:,:] = 4
        return dummy
    if flavor == 3:
        dummy = np.zeros((32,32,32))
        dummy[1:12,1:12,1:12] = ball(5)*1
        dummy[14:31,14:31,14:31] = ball(8)*2
        return dummy

def generate_checkerboard_dummy(board_shape,region_size,region_intensities):
    """Generates a 3D checkerboard dummy."""
    dummy = np.array(region_intensities).reshape(board_shape)
    for axis, size in enumerate(region_size):
        dummy = np.repeat(dummy, size, axis=axis)
    label = np.array(range(len(region_intensities))).reshape(board_shape)
    for axis, size in enumerate(region_size):
        label = np.repeat(label, size, axis=axis)
    return dummy, label

def generate_moons_dummy(radius_1, radius_2, distance, color_1, color_2):
    """Generates two spheres (where 2 overlaps 1)"""
    dummy = np.zeros(((radius_1)+(radius_2)+distance+20, max(radius_1,radius_2)*2+20, max(radius_1,radius_2)*2+20))
    placeholder1, placeholder2 = np.zeros_like(dummy), np.zeros_like(dummy)
    # Computing center of moons:
    center = max(radius_1, radius_2) + 11
    # First moon
    placeholder1[center-radius_1:center+radius_1+1,center-radius_1:center+radius_1+1,center-radius_1:center+radius_1+1] = ball(radius_1)*color_1
    # Second moon
    placeholder2[-(center+radius_2+1):-(center-radius_2),-(center+radius_2+1):-(center-radius_2),-(center+radius_2+1):-(center-radius_2)] = ball(radius_2)*color_2
    dummy[placeholder1==color_1] = color_1
    dummy[placeholder2==color_2] = color_2
    label = np.zeros_like(dummy)
    label[dummy == color_1] = 1
    label[dummy == color_2] = 2
    return dummy, label

def compute_attributes(volume, labelmap, attribute):
    """Computes an specific attribute for an entire volume"""
    if attribute == "centroid":
        labels = np.unique(labelmap)
        centroids = measure_center_of_mass(np.ones_like(labelmap), labels=labelmap, index=labels)
        centroids = np.array(centroids)
        return centroids
    elif attribute == "intensity":
        labels, indexes = np.unique(labelmap, return_inverse=True)
        intensities = np.empty(len(labels))
        for i, label in enumerate(labels):
            intensities[i] = np.mean(volume.flatten()[indexes==i])
        return intensities
    elif attribute == "size":
        labels,voxel_count_per_labels = np.unique(labelmap, return_counts=True)
        sizes = voxel_count_per_labels
        return sizes
    else:
        raise Exception("{} is not a supported attribute".format(attribute))

def build_graph(volume,labelmap,add_edges=True, target_vertices=None):
    """Builds a graph from an annotated volume."""
    # Compute statistical attributes
    centroids = compute_attributes(volume, labelmap, attribute="centroid")
    intensities = compute_attributes(volume, labelmap, attribute="intensity")
    sizes = compute_attributes(volume, labelmap, attribute="size")

    # Assemble statistical attributes as the vertex matrix
    if target_vertices is None:
        vertices = np.column_stack([centroids, intensities, sizes])
    else:
        vertices = np.empty((target_vertices, 5))
        actual_labels = np.unique(labelmap)
        actual_index = 0
        for i, label in enumerate(range(target_vertices)):
            if label in actual_labels:
                vertices[label] = np.append(centroids[actual_index],(intensities[actual_index],sizes[actual_index]))
                actual_index += 1
            else:
                vertices[label] = np.array([math.inf]*5)

    if not add_edges:
        return SRG(vertices, np.array([]), ["centroid_x", "centroid_y", "centroid_z", "intensity", "size"] ,[])
    else:
        # Compute relational attributes
        centroids = vertices[:,:3]
        positions = np.repeat(centroids, centroids.shape[0],axis=0) - np.vstack([centroids]*centroids.shape[0])
        #contrasts = np.repeat(intensities, intensities.shape[0],axis=0) / np.vstack([intensities]*intensities.shape[0])
        #ratios = np.repeat(sizes, sizes.shape[0],axis=0) / np.vstack([sizes]*sizes.shape[0])
        # Assemble relational attributes as the edges matrix
        edges = positions#np.concatenate([positions, contrasts, ratios],axis=-1)

        # Initializing and returning the SRG
        return SRG(vertices, edges, ["centroid_x", "centroid_y", "centroid_z", "intensity", "size"], ["position"])#, "contrast", "ratio"])

if __name__ == '__main__':
    # Step 1: Loading data (generating dummies)
    # -----------------------
    #model_dummy, model_labelmap = generate_dummy(3), generate_dummy_label(3)
    #model_dummy, model_labelmap = generate_checkerboard_dummy((4,4,2), (30,30,30), np.arange(4*4*2)*50)
    model_dummy, model_labelmap = generate_moons_dummy(20,20,3,1,0.5)
    display_volume(model_dummy, cmap="gray", title="Model Input")
    #observation_dummy = generate_dummy(2)
    #observation_dummy = np.random.normal(model_dummy, 20)
    #observation_dummy = random_noise(model_dummy, "s&p", seed=10, amount=0.05)
    observation_dummy = deepcopy(model_dummy)
    display_volume(observation_dummy, cmap="gray", title="Observation Input")
    color_map = ListedColormap([(1,0,0),(0,1,0),(0,0,1),(1,1,0),(1,0,1),(0,1,1)])

    # Step 2: Generating model graph
    # -----------------------
    model_graph = build_graph(model_dummy, model_labelmap)
    print("Model:",represent_srg(model_graph))
    # Normalize the graph? maybe later. note: normalization is a distinct function!

    # Step 3: Generating observation
    # -----------------------
    # Applying gradient
    magnitude = ndi.morphology.morphological_gradient(observation_dummy, 3)
    #display_volume(magnitude, cmap="gray", title="Magnitude")
    # Getting local minima of the volume with a structural element 5x5x1
    volume_local_minima = h_minima(magnitude, h=10)
    # Labeling local_minima
    markers, total_markers = ndi.label(volume_local_minima)
    observed_labelmap = watershed(magnitude,markers=markers)-1
    display_segments_as_lines(observation_dummy, observed_labelmap)
    #display_volume(observed_labelmap)

    # Step 4: Generating super-observation graph
    # -----------------------
    super_graph = build_graph(observation_dummy, observed_labelmap, add_edges=False)
    print("Superobservation:",represent_srg(super_graph))

    # Step 5: Generating initial solution
    # -----------------------
    solution = np.empty(super_graph.vertices.shape[0])
    for i, super_vertex in enumerate(super_graph.vertices):
        # Computing cost to all model vertices
        super_vertex_matrix = np.vstack([super_vertex]*model_graph.vertices.shape[0])
        costs = np.linalg.norm((super_vertex_matrix[:,:-1]-model_graph.vertices[:,:-1]), axis=-1)
        solution[i] = np.argmin(costs)
    print("Inital solution:")
    for i, prediction in enumerate(solution):
        print("\t{}: {}".format(i, prediction))

    # Step 6: Region Joining
    # -----------------------
    joined_labelmap = np.zeros_like(observed_labelmap)
    for element, prediction in enumerate(solution):
        joined_labelmap[observed_labelmap==element]=prediction
    observation_graph = build_graph(observation_dummy, joined_labelmap, target_vertices=model_graph.vertices.shape[0])
    vertex_costs = np.mean(np.linalg.norm(observation_graph.vertices - model_graph.vertices, axis=-1))
    edge_costs = np.mean(np.linalg.norm(observation_graph.edges - model_graph.edges, axis=-1))
    print("Joined Initial Solution (Costs: {:.3f},{:.3f})".format(vertex_costs,edge_costs))
    display_volume(joined_labelmap, cmap=color_map, title="Joined Initial Solution (Costs: {:.3f},{:.3f})".format(vertex_costs,edge_costs))
    print("Observation:",represent_srg(observation_graph))

    # Step 7: Improvement
    # -----------------------
    for epoch in range(1):
        for i, super_vertex in enumerate(super_graph.vertices):
            current_prediction = solution[i]
            current_vertex_costs = np.mean(np.linalg.norm(observation_graph.vertices - model_graph.vertices, axis=-1))
            current_edge_costs = np.mean(np.linalg.norm(observation_graph.edges - model_graph.edges, axis=-1))
            current_cost = current_vertex_costs + current_edge_costs
            # sanity check
            if math.isnan(current_cost): current_cost = math.inf

            for j, potential_prediction in enumerate(model_graph.vertices):
                # Skipping same replacements
                if j == current_prediction: continue
                # Replacing the supervertex's labels
                working_labelmap = deepcopy(joined_labelmap)
                working_labelmap[observed_labelmap==i] = j
                #display_volume(working_labelmap,cmap=color_map, title="Replacing {}'s label (curr: {}) with {}".format(i, solution[i],j))
                # Updating graph
                working_graph = build_graph(observation_dummy, working_labelmap, target_vertices=model_graph.vertices.shape[0])
                #print(represent_srg(working_graph))

                # Computing costs
                potential_vertex_costs = np.mean(np.linalg.norm(working_graph.vertices - model_graph.vertices, axis=-1))
                potential_edge_costs = np.mean(np.linalg.norm(working_graph.edges - model_graph.edges, axis=-1))
                potential_cost = potential_vertex_costs + potential_edge_costs
                print("Replacing {}'s label (curr: {}) with {}".format(i, solution[i],j))
                print("\t cost is {:.2f} (current best: {:.2f})".format(potential_cost,current_cost))
                # Improving if better
                if potential_cost < current_cost:
                    current_prediction = j
                    current_vertex_costs = potential_vertex_costs
                    current_edge_costs = potential_edge_costs
                    current_cost = potential_cost

            # Replacing best in solution
            solution[i] = current_prediction

        # End of an epoch, rebuilding the joined graph
        print("End of epoch #{}: solution = {}".format(epoch,solution))
        joined_labelmap = np.zeros_like(observed_labelmap)
        for element, prediction in enumerate(solution):
            joined_labelmap[observed_labelmap==element]=prediction
        observation_graph = build_graph(observation_dummy, joined_labelmap)
        vertex_costs = np.mean(np.linalg.norm(observation_graph.vertices - model_graph.vertices, axis=-1))
        edge_costs = np.mean(np.linalg.norm(observation_graph.edges - model_graph.edges, axis=-1))
        print("Joined Epoch #{} Solution (Costs: {:.3f},{:.3f})".format(epoch, vertex_costs,edge_costs))
        display_volume(joined_labelmap, cmap=color_map, title="Joined Epoch #{} Solution (Costs: {:.3f},{:.3f})".format(epoch, vertex_costs,edge_costs))
        print("Observation:",represent_srg(observation_graph))


#TODO: separate normalization function. graph-builder using only maps? or directly objects?
#TODO: remember to add norm, header prod, weights...
