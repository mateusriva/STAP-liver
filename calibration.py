"""Functions for calibrating the pipeline with dummies.

This module contains both functions for generating dummies,
and for putting them through the segmentation pipeline."""

import numpy as np
from scipy.ndimage.measurements import center_of_mass as measure_center_of_mass
from time import time
from copy import deepcopy
from itertools import product
from skimage.morphology import watershed, local_minima
from skimage.color import rgb2gray
import scipy.ndimage as ndi
from matplotlib.colors import ListedColormap

from srg import SRG
from display_utils import display_volume, display_segments_as_lines, display_solution
import liver_couinaud as lc

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
        for label in labels:
            intensities[int(label)] = np.mean(volume.flatten()[indexes==label])
        return intensities
    elif attribute == "size":
        labels,voxel_count_per_labels = np.unique(labelmap, return_counts=True)
        sizes = voxel_count_per_labels
        return sizes
    else:
        raise Exception("{} is not a supported attribute".format(attribute))

def compute_attributes_specific(volume, labelmap, specific_label, attribute):
    """Computes an specific attribute for an specific label"""
    if attribute == "centroid":
        centroids = measure_center_of_mass(labelmap==specific_label)
        centroids = np.array(centroids)
        return centroids
    elif attribute == "intensity":
        intensities = np.mean(volume[labelmap==specific_label])
        return intensities
    elif attribute == "size":
        sizes = np.array(np.count_nonzero(labelmap == specific_label))
        return sizes
    else:
        raise Exception("{} is not a supported attribute".format(attribute))

def build_graph(volume,labelmap,add_edges=True):
    """Builds a graph from an annotated volume."""
    # Compute statistical attributes
    centroids = compute_attributes(volume, labelmap, attribute="centroid")
    intensities = compute_attributes(volume, labelmap, attribute="intensity")
    sizes = compute_attributes(volume, labelmap, attribute="size")
    # Assemble statistical attributes as the vertex matrix
    vertices = np.column_stack([centroids, intensities, sizes])

    if not add_edges:
        return SRG(vertices, np.array([]), ["centroid_x", "centroid_y", "centroid_z", "intensity", "size"] ,[])
    else:
        # Compute relational attributes
        positions = np.repeat(centroids, centroids.shape[0],axis=0) - np.vstack([centroids]*centroids.shape[0])
        #contrasts = np.repeat(intensities, intensities.shape[0],axis=0) / np.vstack([intensities]*intensities.shape[0])
        #ratios = np.repeat(sizes, sizes.shape[0],axis=0) / np.vstack([sizes]*sizes.shape[0])
        # Assemble relational attributes as the edges matrix
        edges = positions#np.concatenate([positions, contrasts, ratios],axis=-1)

        # Initializing and returning the SRG
        return SRG(vertices, edges, ["centroid_x", "centroid_y", "centroid_z", "intensity", "size"], ["position"])#, "contrast", "ratio"])

def rebuild_graph(base_graph, volume,labelmap,old,new,add_edges=True):
    """Rebuilds a graph from a modification of an annotation."""
    # Copy attributes
    new_SRG = deepcopy(base_graph)
    centroids = new_SRG.vertices[:,:3]
    intensities = new_SRG.vertices[:,3]
    sizes = new_SRG.vertices[:,4]
    print(centroids, intensities, sizes)
    # Compute statistical attributes
    #BUG: how to solve disappearing vertices??
    centroids[old] = compute_attributes_specific(volume, labelmap, old, attribute="centroid")
    intensities[old] = compute_attributes_specific(volume, labelmap, old, attribute="intensity")
    sizes[old] = compute_attributes_specific(volume, labelmap, old, attribute="size")
    centroids[new] = compute_attributes_specific(volume, labelmap, new, attribute="centroid")
    intensities[new] = compute_attributes_specific(volume, labelmap, new, attribute="intensity")
    sizes[new] = compute_attributes_specific(volume, labelmap, new, attribute="size")
    # Assemble statistical attributes as the vertex matrix

    print(centroids, intensities, sizes)
    new_SRG.vertices = np.column_stack([centroids, intensities, sizes])

    if not add_edges:
        return new_SRG
    else:
        # Compute relational attributes
        positions = np.repeat(centroids, centroids.shape[0],axis=0) - np.vstack([centroids]*centroids.shape[0])
        #contrasts = np.repeat(intensities, intensities.shape[0],axis=0) / np.vstack([intensities]*intensities.shape[0])
        #ratios = np.repeat(sizes, sizes.shape[0],axis=0) / np.vstack([sizes]*sizes.shape[0])
        # Assemble relational attributes as the edges matrix
        new_SRG.edges = positions#np.concatenate([positions, contrasts, ratios],axis=-1)

        # Initializing and returning the SRG
        return new_SRG


if __name__ == '__main__':
    # Step 1: Loading data (generating dummies)
    # -----------------------
    model_dummy, model_labelmap = generate_dummy(0), generate_dummy_label(0)
    #display_volume(model_dummy, cmap="gray", "Model Input")
    observation_dummy = generate_dummy(2)
    #display_volume(observation_dummy, cmap="gray", title="Observation Input")
    color_map = ListedColormap([(1,0,0),(0,1,0),(0,0,1),(1,1,0),(1,0,1),(0,1,1)])

    # Step 2: Generating model graph
    # -----------------------
    model_graph = build_graph(model_dummy, model_labelmap)
    print("Model:",lc.represent_liver_srg(model_graph))
    # Normalize the graph? maybe later. note: normalization is a distinct function!

    # Step 3: Generating observation
    # -----------------------
    # Applying gradient
    magnitude = ndi.morphology.morphological_gradient(observation_dummy, 3)
    #display_volume(magnitude, cmap="gray", title="Magnitude")
    # Getting local minima of the volume with a structural element 5x5x1
    volume_local_minima = local_minima(magnitude)
    # Labeling local_minima
    markers, total_markers = ndi.label(volume_local_minima)
    observed_labelmap = watershed(magnitude,markers=markers)-1
    #display_segments_as_lines(observation_dummy, observed_labelmap)
    #display_volume(observed_labelmap)

    # Step 4: Generating super-observation graph
    # -----------------------
    super_graph = build_graph(observation_dummy, observed_labelmap, add_edges=False)
    print("Superobservation:",lc.represent_liver_srg(super_graph))

    # Step 5: Generating initial solution
    # -----------------------
    solution = np.empty(super_graph.vertices.shape[0])
    for i, super_vertex in enumerate(super_graph.vertices):
        # Computing cost to all model vertices
        super_vertex_matrix = np.vstack([super_vertex]*model_graph.vertices.shape[0])
        costs = np.linalg.norm((super_vertex_matrix-model_graph.vertices), axis=-1)
        solution[i] = np.argmin(costs)
    print("Inital solution:")
    for i, prediction in enumerate(solution):
        print("\t{}: {}".format(i, prediction))

    # Step 6: Region Joining
    # -----------------------
    joined_labelmap = np.zeros_like(observed_labelmap)
    for element, prediction in enumerate(solution):
        joined_labelmap[observed_labelmap==element]=prediction
    observation_graph = build_graph(observation_dummy, joined_labelmap)
    vertex_costs = np.mean(np.linalg.norm(observation_graph.vertices - model_graph.vertices, axis=-1))
    edge_costs = np.mean(np.linalg.norm(observation_graph.edges - model_graph.edges, axis=-1))
    print("Joined Initial Solution (Costs: {:.3f},{:.3f})".format(vertex_costs,edge_costs))
    display_volume(joined_labelmap, cmap=color_map, title="Joined Initial Solution (Costs: {:.3f},{:.3f})".format(vertex_costs,edge_costs))

    # Step 7: Improvement
    # -----------------------
    for epoch in range(1):
        for i, super_vertex in enumerate(super_graph.vertices):
            current_prediction = solution[i]
            current_vertex_costs = np.mean(np.linalg.norm(observation_graph.vertices - model_graph.vertices, axis=-1))
            current_edge_costs = np.mean(np.linalg.norm(observation_graph.edges - model_graph.edges, axis=-1))
            current_cost = current_vertex_costs + current_edge_costs

            for j, potential_prediction in enumerate(model_graph.vertices):
                # Skipping same replacements
                if j == current_prediction: continue
                # Replacing the supervertex's labels
                working_labelmap = deepcopy(joined_labelmap)
                working_labelmap[observed_labelmap==i] = j
                display_volume(working_labelmap,cmap=color_map, title="Replacing {}'s label (curr: {}) with {}".format(i, solution[i],j))
                # Updating graph
                working_graph = build_graph(observation_dummy, working_labelmap)

                # Computing costs
                potential_vertex_costs = np.mean(np.linalg.norm(observation_graph.vertices - model_graph.vertices, axis=-1))
                potential_edge_costs = np.mean(np.linalg.norm(observation_graph.edges - model_graph.edges, axis=-1))
                potential_cost = potential_vertex_costs + potential_edge_costs
                # Improving if better
                if potential_cost < current_cost:
                    current_prediction = j
                    current_vertex_costs = potential_vertex_costs
                    current_edge_costs = potential_edge_costs
                    current_cost = potential_cost

            # Replacing best in solution
            solution[i] = current_prediction

        # End of an epoch, rebuilding the joined graph
        joined_labelmap = np.zeros_like(observed_labelmap)
        for element, prediction in enumerate(solution):
            joined_labelmap[observed_labelmap==element]=prediction
        observation_graph = build_graph(observation_dummy, joined_labelmap)
        vertex_costs = np.mean(np.linalg.norm(observation_graph.vertices - model_graph.vertices, axis=-1))
        edge_costs = np.mean(np.linalg.norm(observation_graph.edges - model_graph.edges, axis=-1))
        print("Joined Epoch #{} Solution (Costs: {:.3f},{:.3f})".format(epoch, vertex_costs,edge_costs))
        display_volume(joined_labelmap, cmap=color_map, title="Joined Epoch #{} Solution (Costs: {:.3f},{:.3f})".format(epoch, vertex_costs,edge_costs))



#TODO: separate normalization function. graph-builder using only maps? or directly objects?
#TODO: remember to add norm, header prod, weights...
