"""Full Liver segmentation module for the SRG.

This module contains specific configurations
for the SRG, in order to make it segment livers.

Authors:
 * Mateus Riva (mriva@ime.usp.br)
"""

import numpy as np
import math
from scipy.ndimage.measurements import center_of_mass as measure_center_of_mass
from time import time
from copy import deepcopy
from itertools import product
from skimage.morphology import watershed, local_minima
from skimage.color import rgb2gray
import scipy.ndimage as ndi
from matplotlib.colors import ListedColormap

from srg import SRG
from patient import Patient, LabelMap, Volume
from display_utils import display_volume, display_segments_as_lines, display_solution, represent_srg

"""Display definitions"""
class_names = ["BG Posterior","BG Anterior","BG OtherBody","Vena Cava","Portal Vein","Left H. Vein","Middle H. Vein","Right H. Vein","Liver"]
class_colors = ListedColormap([(0,0,0),(0.5,0.5,0.5),(1,1,1),(0,0,1),(0,1,1),(0.5,0,1),(0,0.5,1),(0,0,0.5),(1,0,0)])

def compute_attributes(volume, labelmap, attribute):
    """Computes an specific attribute for an entire volume"""
    if attribute == "centroid":
        labels = np.unique(labelmap.data)
        centroids = measure_center_of_mass(np.ones_like(labelmap.data), labels=labelmap.data, index=labels)
        centroids = np.array(centroids)
        centroids *= volume.header["spacings"]
        return centroids
    elif attribute == "intensity":
        labels, indexes = np.unique(labelmap.data, return_inverse=True)
        intensities = np.empty(len(labels))
        for i, label in enumerate(labels):
            intensities[i] = np.mean(volume.data.flatten()[indexes==i])
        return intensities
    elif attribute == "size":
        labels,voxel_count_per_labels = np.unique(labelmap.data, return_counts=True)
        sizes = voxel_count_per_labels * np.prod(volume.header["spacings"])
        return sizes
    else:
        raise Exception("{} is not a supported attribute".format(attribute))

def build_graph(volume,labelmap,add_edges=True, target_vertices=None):
    """Builds a graph from an annotated volume.

    Parameters
    ----------
    volume: `obj:Volume`
        A patient's Volume object.
    labelmap: `obj:LabelMap`
        A patient's LabelMap object.
    add_edges: bool
        If False, the generated graph will not have edges.
    target_vertices : int or `None`
        If not None, then the final SRG will have at least `target_vertices` vertices."""
    # Compute statistical attributes
    centroids = compute_attributes(volume, labelmap, attribute="centroid")
    intensities = compute_attributes(volume, labelmap, attribute="intensity")
    sizes = compute_attributes(volume, labelmap, attribute="size")

    # Assemble statistical attributes as the vertex matrix
    if target_vertices is None:
        vertices = np.column_stack([centroids, intensities, sizes])
    else:
        vertices = np.empty((target_vertices, 5))
        actual_labels = np.unique(labelmap.data)
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
    # Step 1: Loading data
    # -----------------------
    print("# Step 1: Loading data")
    model_patient = Patient.build_from_folder("data/4")
    model_volume, model_labelmap = model_patient.volumes['t2'], model_patient.labelmaps['t2']
    # Reconfiguring model_labelmap with extra backgrounds and unified liver
    model_labelmap.data += 2 # Adding space for the extra labels at the start
    model_labelmap.data[np.logical_and(model_volume.data < 10, model_labelmap.data == 2)] = 0 # posterior background is 0
    model_labelmap.data[model_labelmap.data.shape[1]//2:,:,:][model_labelmap.data[model_labelmap.data.shape[1]//2:,:,:] == 0] = 1 # anterior background is 1
    model_labelmap.data[model_labelmap.data >= 8] = 8
    model_labelmap.header["num_labels"] = 9

    observation_volume = deepcopy(model_volume)

    # Step 2: Generating model graph
    # -----------------------
    print("# Step 2: Generating model graph")
    model_graph = build_graph(model_volume, model_labelmap)
    print("Model:",represent_srg(model_graph, class_names=class_names))
    # Normalize the graph? maybe later. note: normalization is a distinct function!

    # Step 3: Generating observation
    # -----------------------
    print("# Step 3: Generating observation")
    # Filtering
    filtered_volume=ndi.gaussian_filter(observation_volume.data, (5,5,1))
    filtered_volume[filtered_volume < 10] = 0
    filtered_volume = filtered_volume / np.max(filtered_volume)
    # Applying gradient
    magnitude = ndi.morphology.morphological_gradient(filtered_volume, (19,19,5))
    #display_volume(magnitude, cmap="gray", title="Magnitude")
    # Getting local minima of the volume with a structural element 5x5x1
    volume_local_minima = local_minima(magnitude, selem=np.ones((5,5,5)))
    # Labeling local_minima
    markers, total_markers = ndi.label(volume_local_minima)
    observed_labelmap = LabelMap("super",None,watershed(magnitude,markers=markers)-1)
    #display_segments_as_lines(observation_volume.data, observed_labelmap.data)
    #display_volume(observed_labelmap.data)

    # Step 4: Generating super-observation graph
    # -----------------------
    print("# Step 4: Generating super-observation graph")
    super_graph = build_graph(observation_volume, observed_labelmap, add_edges=False)
    #print("Superobservation:",represent_srg(super_graph, vertex_range=(100,190)))

    # Step 5: Generating initial solution
    # -----------------------
    print("# Step 5: Generating initial solution")
    #TODO: PROPER WEIGHTSSSS
    solution = np.empty(super_graph.vertices.shape[0])
    for i, super_vertex in enumerate(super_graph.vertices):
        # Computing cost to all model vertices
        super_vertex_matrix = np.vstack([super_vertex]*model_graph.vertices.shape[0])
        costs = np.linalg.norm((0.3,0.3,0.3,0.1)*(super_vertex_matrix[:,:-1]-model_graph.vertices[:,:-1]), axis=-1)
        solution[i] = np.argmin(costs)

    # Step 6: Region Joining
    # -----------------------
    print("# Step 6: Region Joining")
    joined_labelmap = LabelMap("joined", None, np.zeros_like(observed_labelmap.data))
    for element, prediction in enumerate(solution):
        joined_labelmap.data[observed_labelmap.data==element]=prediction
    observation_graph = build_graph(observation_volume, joined_labelmap, target_vertices=model_graph.vertices.shape[0])
    vertex_costs = np.mean(np.linalg.norm(observation_graph.vertices - model_graph.vertices, axis=-1))
    edge_costs = np.mean(np.linalg.norm(observation_graph.edges - model_graph.edges, axis=-1))
    print("Joined Initial Solution (Costs: {:.3f},{:.3f})".format(vertex_costs,edge_costs))
    #display_volume(joined_labelmap.data, cmap=class_colors, title="Joined Initial Solution (Costs: {:.3f},{:.3f})".format(vertex_costs,edge_costs))
    print("Observation:",represent_srg(observation_graph, class_names=class_names))

    # Step 7: Improvement
    # -----------------------
    print("# Step 7: Improvement")
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
                working_labelmap.data[observed_labelmap==i] = j
                #display_volume(working_labelmap,cmap=class_colors, title="Replacing {}'s label (curr: {}) with {}".format(i, solution[i],j))
                # Updating graph
                working_graph = build_graph(observation_volume, working_labelmap, target_vertices=model_graph.vertices.shape[0])

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
        joined_labelmap = LabelMap("joined", None, np.zeros_like(observed_labelmap.data))
        for element, prediction in enumerate(solution):
            joined_labelmap.data[observed_labelmap.data==element]=prediction
        observation_graph = build_graph(observation_volume, joined_labelmap, target_vertices=model_graph.vertices.shape[0])
        vvertex_costs = np.mean(np.linalg.norm(observation_graph.vertices - model_graph.vertices, axis=-1))
        edge_costs = np.mean(np.linalg.norm(observation_graph.edges - model_graph.edges, axis=-1))
        print("Joined Epoch #{} Solution (Costs: {:.3f},{:.3f})".format(epoch, vertex_costs,edge_costs))
        print("Observation:",represent_srg(observation_graph))
        display_volume(joined_labelmap.data, cmap=class_colors, title="Joined Epoch #{} Solution (Costs: {:.3f},{:.3f})".format(epoch, vertex_costs,edge_costs))


#TODO: separate normalization function. graph-builder using only maps? or directly objects?
#TODO: remember to add norm, header prod, weights...
