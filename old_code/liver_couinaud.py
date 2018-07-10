"""Liver segmentation module for the SRG.

This module contains specific configurations
for the SRG, in order to make it segment liver
substructures.

Authors:
 * Mateus Riva (mriva@ime.usp.br)
"""

import pickle, sys

import numpy as np
from patient import Patient
from srg import SRG
from display_utils import display_volume
from scipy.ndimage.measurements import center_of_mass as measure_center_of_mass
from matplotlib.colors import ListedColormap

class_names=["BG Posterior","BG Anterior","BG OtherBody","Vena Cava","Portal Vein","Left H. Vein","Middle H. Vein","Right H. Vein","Segment I","Segment II","Segment III","Segment IVa","Segment IVb","Segment V","Segment VI","Segment VII","Segment VIII"]
"""Name of each model class."""
class_colors=[(0,0,0),(0.5,0.5,0.5),(1,1,1),(0,0,1),(0,1,1),(0.5,0,1),(0,0.5,1),(0,0,0.5),(1,0,0),(1,0.5,0),(0.5,0.5,0),(0,1,0),(0,0.5,0),(1,0.5,1),(0.5,0,0.5),(1,1,0),(1,0,1)]
"""Color of each model class."""

def compute_centroids(volume, labelmap, specific_label=None):
    """Computes centroids for each label in a volume.

    Returns
    -------
    centroids : `2darray`
        Array with `num_labels` lines and 3 columns. Each line contains
        the normalized, real x, y, and z of each label.
    """
    # get sorted labels and center-of-mass for each label
    if specific_label is None:
        labels = np.unique(labelmap.data)
        centroids = measure_center_of_mass(np.ones_like(labelmap.data), labels=labelmap.data, index=labels)
    else: # specific label only
        centroids = measure_center_of_mass(labelmap.data==specific_label)
    centroids = np.array(centroids)

    # multiply voxel value by voxel size to get real centroid
    centroids *= volume.header["spacings"]

    return centroids

def compute_intensities(volume, labelmap, specific_label=None):
    """Computes mean intensities for each label in a volume.

    TODO: some volume normalization? Gradient, maybe? Or mean norm of voxels?

    Returns
    -------
    intensities : `2darray`
        Array with `num_labels` lines and 1 column. Each line contains
        the absolute mean intensity of each label.
    """
    if specific_label is None:
        # get sorted labels and indexes for each label
        labels, indexes = np.unique(labelmap.data, return_inverse=True)
        # initializing intensities array
        intensities = np.empty((len(labels),1))
        # building mean intensity attribute list
        for label in labels:
            intensities[label] = np.mean(volume.data.flatten()[indexes==label])
    else: # specific label only
        intensities = np.mean(volume.data[labelmap.data==specific_label])

    return intensities

def compute_sizes(volume, labelmap, specific_label=None):
    """Computes the size (volume) of each label in a volume.

    Returns
    -------
    sizes : `2darray`
        Array with `num_labels` lines and 1 column. Each line contains
        the absolute size of each label.
    """
    if specific_label is None:
        # get sorted labels and voxel count per label
        labels,voxel_count_per_labels = np.unique(labelmap.data, return_counts=True)
        # multiplying by the real size of the voxel
        sizes = voxel_count_per_labels * np.prod(volume.header["spacings"])
        sizes = sizes.reshape((voxel_count_per_labels.shape[0],1))
    else: # specific label only
        sizes = np.count_nonzero(labelmap.data == specific_label)*np.prod(volume.header["spacings"])

    return sizes

def liver_build_from_patient(patient, vertex_mean=None,vertex_std=None,edge_mean=None,edge_std=None, return_stats=False):
    """Assembles a liver SRG from an annotated patient.

    Builds a fully-connected SRG from an annotated patient,
    for liver subsegmentation, utilizing the following attributes:
    * Structural: real centroid, absolute intensity, real volume, #semi-major axis
    * Relational: real centroid vector, #relative intensity, relative volume, #relative orientation

    Parameters
    ----------
    patient : `obj:Patient`
        An annotated Patient object. Note that this method
        currently supports t2 patients only.

    Returns
    -------
    srg : `obj:SRG`
        Structural-relational graph corresponding to the
        input Patient.
    """
    # Acquiring t2 volume and labelmap
    volume = patient.volumes['t2']
    labelmap = patient.labelmaps['t2']

    # Compute statistical attributes
    centroids = compute_centroids(volume, labelmap)
    intensities = compute_intensities(volume, labelmap)
    sizes = compute_sizes(volume, labelmap)
    # Assemble statistical attributes as the vertex matrix
    vertices = np.concatenate([centroids, intensities, sizes],axis=-1)
    # Normalizing to normal~(0,1)
    if vertex_mean is None: vertex_mean = vertices.mean(axis=0)
    if vertex_std is None: vertex_std = vertices.std(axis=0)
    vertices = (vertices - vertex_mean) / vertex_std

    # Compute relational attributes
    positions = np.repeat(centroids, centroids.shape[0],axis=0) - np.vstack([centroids]*centroids.shape[0])
    #contrasts = np.repeat(intensities, intensities.shape[0],axis=0) / np.vstack([intensities]*intensities.shape[0])
    #ratios = np.repeat(sizes, sizes.shape[0],axis=0) / np.vstack([sizes]*sizes.shape[0])
    # Assemble relational attributes as the edges matrix
    edges = positions#np.concatenate([positions, contrasts, ratios],axis=-1)
    # Normalizing to normal~(0,1)
    if edge_mean is None: edge_mean = edges.mean(axis=0)
    if edge_std is None: edge_std = edges.std(axis=0)
    edges = (edges - edge_mean) / edge_std

    # Initializing and returning the SRG
    new_SRG = SRG(vertices, edges, ["centroid_x", "centroid_y", "centroid_z", "intensity", "size"], ["position"])#, "contrast", "ratio"])
    if return_stats:
        return new_SRG, vertex_mean, vertex_std, edge_mean, edge_std
    else:
        return new_SRG

def represent_liver_srg(object_graph,is_model=False,vertex_range=None):
    """Returns a liver SRG as an human readable string.
    """
    representation = str(object_graph)
    if vertex_range is None:
        represent_vertices = object_graph.vertices
    else:
        represent_vertices = object_graph.vertices[vertex_range[0]:vertex_range[1]]

    representation += "\n#\tClasses\t"
    for attr in object_graph.vertex_attributes:
        representation += "\t|{}".format(attr)
    representation += "\n"
    for i, vertex in enumerate(represent_vertices):
        if is_model:
            representation += "{}\t{}".format(i,class_names[i])
        else:
            representation += "{}\t\t".format(i)
        for j in range(len(object_graph.vertex_attributes)):
            representation += "\t|{:.3f}\t".format(vertex[j])
        representation += "\n"

    return representation

def compute_cost(x, y, weights=None):
    """Computes the row-wise cost between two matrixes."""
    if weights is None:
        if len(x.shape) > 1:
            weights = np.ones_like(x.shape[1])
        else:
            weights = np.ones_like(x.shape[0])
    weights = weights/np.sum(weights)
    return np.linalg.norm(weights*(x-y), axis=-1)

def update_graph(graph, old, new, patient, vertex_mean=None,vertex_std=None,edge_mean=None,edge_std=None):
    """Updates a graph which changed two labels.

    Parameters/
    ----------
    graph : `SRG`
        Graph to be updated.
    old : `int`
        Label that was replaced by `new`.
    new : `int`
        Label that replaced `old`.
    patient : `Patient`
        Patient object with updated labelmap.
    """
    # Acquiring t2 volume and labelmap
    volume = patient.volumes['t2']
    labelmap = patient.labelmaps['t2']

    # Compute statistical attributes
    updated_vertices = (graph.vertices*vertex_std) + vertex_mean
    centroids = graph.vertices[:,:3]
    intensities = graph.vertices[:,3]
    sizes = graph.vertices[:,4]

    centroids[old] = compute_centroids(volume, labelmap, old)
    intensities[old] = compute_intensities(volume, labelmap, old)
    sizes[old] = compute_sizes(volume, labelmap, old)

    centroids[new] = compute_centroids(volume, labelmap, new)
    intensities[new] = compute_intensities(volume, labelmap, new)
    sizes[new] = compute_sizes(volume, labelmap, new)

    # Assemble statistical attributes as the vertex matrix
    vertices = np.concatenate([centroids, intensities.reshape(intensities.shape[0],1), sizes.reshape(intensities.shape[0],1)],axis=-1)
    # Normalizing to normal~(0,1)
    if vertex_mean is None: vertex_mean = vertices.mean(axis=0)
    if vertex_std is None: vertex_std = vertices.std(axis=0)
    vertices = (vertices - vertex_mean) / vertex_std

    # Compute relational attributes
    positions = np.repeat(centroids, centroids.shape[0],axis=0) - np.vstack([centroids]*centroids.shape[0])
    #contrasts = np.repeat(intensities, intensities.shape[0],axis=0) / np.vstack([intensities]*intensities.shape[0])
    #ratios = np.repeat(sizes, sizes.shape[0],axis=0) / np.vstack([sizes]*sizes.shape[0])
    # Assemble relational attributes as the edges matrix
    edges = positions#np.concatenate([positions, contrasts, ratios],axis=-1)
    # Normalizing to normal~(0,1)
    if edge_mean is None: edge_mean = edges.mean(axis=0)
    if edge_std is None: edge_std = edges.std(axis=0)
    edges = (edges - edge_mean) / edge_std

    return SRG(vertices, edges, ["centroid_x", "centroid_y", "centroid_z", "intensity", "size"], ["position"])#, "contrast", "ratio"])

if __name__ == '__main__':
    from time import time
    from copy import deepcopy
    from itertools import product
    from skimage.morphology import watershed, local_minima
    from skimage.color import rgb2gray
    import scipy.ndimage as ndi
    from display_utils import display_segments_as_lines, display_solution

    print("Loading a single patient... ", end="", flush=True)
    t0 = time()
    model_patient = Patient.build_from_folder("data/4")
    print("Done. {:.4f}s".format(time()-t0))

    # Splitting the background into 3 labels
    model_patient.labelmaps["t2"].data += 2 # Adding space for the extra labels at the start
    model_patient.labelmaps["t2"].data[np.logical_and(model_patient.volumes["t2"].data < 10, model_patient.labelmaps["t2"].data == 2)] = 0 # posterior background is 0
    model_patient.labelmaps["t2"].data[model_patient.labelmaps["t2"].data.shape[1]//2:,:,:][model_patient.labelmaps["t2"].data[model_patient.labelmaps["t2"].data.shape[1]//2:,:,:] == 0] = 1 # anterior background is 1
    model_patient.labelmaps["t2"].header["num_labels"] += 2

    print("Building model graph... ", end="", flush=True)
    t0 = time()
    model_graph, vertex_mean, vertex_std, edge_mean, edge_std = liver_build_from_patient(model_patient, return_stats=True)
    print("Done. {:.4f}s".format(time()-t0))

    #print(represent_liver_srg(model_graph,True))

    # Self-matching for calibration
    print("Calibrating model... ", end="", flush=True)
    t0 = time()
    weights_list = list(product([0,1,2], repeat=4))[1:]
    errors = 0
    for weights in weights_list:
        other_graph = deepcopy(model_graph)
        solution = np.empty(other_graph.vertices.shape[0])
        # Getting closest model vertex, for each vertex
        for i, vertex in enumerate(other_graph.vertices):
            vertex_matrix = np.vstack([vertex]*model_graph.vertices.shape[0])
            distances = compute_cost(model_graph.vertices[:,:-1],vertex_matrix[:,:-1], weights)#np.sum(abs(model_graph.vertices-vertex_matrix),axis=-1)
            solution[i] = np.argmin(distances)
        errors += np.sum(solution != np.arange(model_graph.vertices.shape[0]))
    print("Done. {:.4f}s. {} errors found".format(time()-t0, errors))

    print("Observing image... ", end="", flush=True)
    t0 = time()
    observed_patient = deepcopy(model_patient)
    #observed_labels = slic(observed_patient.volumes['t2'].data, n_segments=400,
    #                    compactness=0.0001, multichannel=False, sigma=(3,3,1))
    # Filtering volume
    filtered_volume=ndi.gaussian_filter(observed_patient.volumes['t2'].data, (3,3,1))
    filtered_volume[filtered_volume < 10] = 0
    filtered_volume = filtered_volume / np.max(filtered_volume)
    # Applying gradient
    magnitude = ndi.morphology.morphological_gradient(filtered_volume, (19,19,5))
    # Getting local minima of the volume with a structural element 5x5x1
    volume_local_minima = local_minima(magnitude, selem=np.ones((5,5,5)))
    # Labeling local_minima
    markers, total_markers = ndi.label(volume_local_minima)
    observed_labels = watershed(magnitude,markers=markers)-1
    print("Done. {:.4f}s".format(time()-t0))
    #display_segments_as_lines(observed_patient.volumes['t2'].data, observed_labels)
    #display_volume(observed_labels, cmap=ListedColormap(np.random.rand(2000,3)))

    print("Building super-observation graph... ", end="", flush=True)
    t0 = time()
    observed_patient.labelmaps["t2"].data = observed_labels
    super_observation_graph = liver_build_from_patient(observed_patient, vertex_mean, vertex_std, edge_mean, edge_std)
    print("Done. {:.4f}s".format(time()-t0))

    initial_weights = [1,1,1,0.5] # Determined subjectively and experimentally - must formalize!

    print("Assembling initial prediction with weights {}... ".format(initial_weights), end="", flush=True)
    t0 = time()
    solution = np.empty(super_observation_graph.vertices.shape[0], dtype=int)
    # Getting closest model vertex, for each vertex
    for i, vertex in enumerate(super_observation_graph.vertices):
        vertex_matrix = np.vstack([vertex]*model_graph.vertices.shape[0])
        distances = compute_cost(model_graph.vertices[:,:-1],vertex_matrix[:,:-1], initial_weights)
        solution[i] = np.argmin(distances)
    print("Done. {:.4f}s".format(time()-t0))
    #display_solution(observed_labels, solution, cmap=ListedColormap(class_colors))

    # Joining regions into an observation graph
    print("Building joined observation graph... ", end="", flush=True)
    t0 = time()
    joined_labels = np.zeros_like(observed_labels)
    for element, prediction in enumerate(solution):
        joined_labels[observed_labels==element]=prediction
    t1 = time()
    joined_patient = deepcopy(observed_patient)
    joined_patient.labelmaps["t2"].data = joined_labels
    observation_graph = liver_build_from_patient(joined_patient, vertex_mean, vertex_std, edge_mean, edge_std)
    print("Done. {:.4f}s + {:.4f}s".format(t1-t0,time()-t1))

    print("Model:",represent_liver_srg(model_graph, True))
    print("Obser:",represent_liver_srg(observation_graph,True))

    # Computing total cost of solution
    vertex_weights, edge_weights=[1,1,1,1,1],[1,1,1]
    # TODO: improve the edge position attribute cost. currently it's just distance.
    print("Computing cost... ", end="", flush=True)
    t0 = time()
    current_vertex_costs = compute_cost(observation_graph.vertices, model_graph.vertices, vertex_weights)
    current_edge_costs = compute_cost(observation_graph.edges, model_graph.edges, edge_weights)
    print("Done. {:.4f}s".format(time()-t0))
    print("Costs are:\n   Total\t| Mean\nV  {:.3f}\t| {:.3f}\nE  {:.3f}\t| {:.3f}".format(np.sum(current_vertex_costs), np.mean(current_vertex_costs),np.sum(current_edge_costs), np.mean(current_edge_costs)))

    # Attempting greedy improvement
    # * Rough algorithm: for each vertex in super-obs, * attempt to change its prediction
    # * Update joined_labels, * recompute ONLY? affected vertices/edges, * check cost
    for epoch in range(1):
        working_solution = deepcopy(solution) # This solution will update at end of epoch
        for i, super_vertex in enumerate(super_observation_graph.vertices):
            # Attempting to change the supervertex's prediction (which is currently solution[i])
            t0 = time()
            print(represent_liver_srg(observation_graph))
            current_vertex_costs = np.mean(compute_cost(observation_graph.vertices, model_graph.vertices, vertex_weights))
            current_edge_costs = np.mean(compute_cost(observation_graph.edges, model_graph.edges, edge_weights))
            current_prediction = solution[i]
            print("Attempting to improve on supervertex {} (currently {})".format(i, solution[i]))
            print("Current costs are {:.3f}, {:.3f} ({:.4f}s)".format(current_vertex_costs,current_edge_costs, time()-t0))

            working_patient = deepcopy(joined_patient)
            for j, potential_prediction in enumerate(model_graph.vertices):
                # updating joined_labels
                working_labels = deepcopy(joined_labels)
                working_labels[observed_labels==i] = j

                # recomputing affected parts of the graph
                working_patient.labelmaps["t2"].data = working_labels
                working_graph = update_graph(observation_graph, solution[i], j, working_patient, vertex_mean, vertex_std, edge_mean, edge_std)

                # computing and comparing new costs
                potential_vertex_costs = np.mean(compute_cost(working_graph.vertices, model_graph.vertices, vertex_weights))
                potential_edge_costs = np.mean(compute_cost(working_graph.edges, model_graph.edges, edge_weights))
                if potential_vertex_costs + potential_edge_costs < current_vertex_costs + current_edge_costs:
                    print("\t{} is better! Costs {:.3f}, {:.3f} ({:.4f})".format(j, potential_vertex_costs, potential_edge_costs, time()-t0))
                    current_vertex_costs = potential_vertex_costs
                    current_edge_costs = potential_edge_costs
                    current_prediction = j

            # Updating prediction
            working_solution[i] = current_prediction

    # Attempting various weights
    # costs, accuracies = [], []
    # weights_list = list(product([1,2], repeat=4))
    # for weights in weights_list:
    #     print("Assembling initial prediction with weights {}... ".format(weights), end="", flush=True)
    #     t0 = time()
    #     solution = np.empty(super_observation_graph.vertices.shape[0])
    #     # Getting closest model vertex, for each vertex
    #     for i, vertex in enumerate(super_observation_graph.vertices):
    #         vertex_matrix = np.vstack([vertex]*model_graph.vertices.shape[0])
    #         distances = compute_cost(model_graph.vertices[:,:-1],vertex_matrix[:,:-1], weights)#np.sum(abs(model_graph.vertices-vertex_matrix),axis=-1)
    #         solution[i] = np.argmin(distances)
    #     print("Done. {:.4f}s".format(time()-t0))
    #
    #     # Displaying a prediction
    #     display_solution(observed_labels, solution, title="Weights: {}".format(weights),cmap=ListedColormap(np.array(class_colors)[np.unique(solution).astype(int)]))
    #     # Computing total prediction cost
    #     cost = np.sum([compute_cost(model_graph.vertices[int(solution[x]),:-1],super_observation_graph.vertices[x,:-1], weights) for x in range(len(super_observation_graph.vertices))])
    #     costs.append(cost)
    #     predicted_labelmap = np.zeros_like(observed_labels)
    #     for element, prediction in enumerate(solution):
    #         predicted_labelmap[observed_labels==element]=prediction
    #     accuracy = np.sum([predicted_labelmap==model_patient.labelmaps['t2'].data]) / predicted_labelmap.size
    #     accuracies.append(accuracy)
    #
    # for i, cost in sorted(enumerate(costs),key=lambda x:x[1]):
    #     print("W {} : Accuracy: {:.1f}%, Cost: {:.3f}".format(weights_list[i],accuracies[i]*100,cost))

    # weights = [1,2,2,2]
    # print("Assembling initial prediction with weights {}... ".format(weights), end="", flush=True)
    # t0 = time()
    # solution = np.empty(super_observation_graph.vertices.shape[0])
    # # Getting closest model vertex, for each vertex
    # for i, vertex in enumerate(super_observation_graph.vertices):
    #     vertex_matrix = np.vstack([vertex]*model_graph.vertices.shape[0])
    #     distances = compute_cost(model_graph.vertices[:,:-1],vertex_matrix[:,:-1], weights)
    #     solution[i] = np.argmin(distances)
    # print("Done. {:.4f}s".format(time()-t0))
    # display_solution(observed_labels, solution, cmap=ListedColormap(class_colors))
    # solution = solution.astype(int)
    # _, super_count = np.unique(solution, return_counts=True)
    #
    # # Joining regions into an observation graph
    # print("Building joined observation graph... ", end="", flush=True)
    # t0 = time()
    # joined_labels = np.zeros_like(observed_labels)
    # for element, prediction in enumerate(solution):
    #     joined_labels[observed_labels==element]=prediction
    # t1 = time()
    # joined_patient = deepcopy(observed_patient)
    # joined_patient.labelmaps["t2"].data = joined_labels
    # observation_graph = liver_build_from_patient(joined_patient, vertex_mean, vertex_std, edge_mean, edge_std)
    # print("Done. {:.4f}s + {:.4f}s".format(t1-t0,time()-t1))
    #
    # print(represent_liver_srg(observation_graph,True))
    #
    # # Greedy improvement of joined observation graph (by changing the super observation graph
    # for epoch in range(4):
    #     # Computing current costs
    #     current_vertex_costs = compute_cost(observation_graph.vertices, model_graph.vertices)
    #     current_edge_costs = compute_cost(observation_graph.edges, model_graph.edges)
    #     print("Starting epoch #{}. Costs are: V={:.2f}; E={:.2f}".format(epoch,np.sum(current_vertex_costs), np.sum(current_edge_costs)))
    #
    #     # Attempting to replace every super-observed region
    #     for i,vertex in enumerate(super_observation_graph.vertices):
    #         best_pred = solution[i]
    #         best_cost = compute_cost(vertex,model_graph.vertices[solution[i]])
    #         for j,potential_prediction in enumerate(model_graph.vertices):
    #             # reassessing affected vertices and edges (TODO: edges how?)
    #             observation_graph.vertices[solution[i],:] += vertex / super_count[solution[i]]
    #             potential_cost = compute_cost(vertex,potential_prediction)
    #             if potential_cost < best_cost:
    #                 best_cost = potential_cost
    #                 best_pred = j
    #
    #         #TODO: change only in end of epoch?
    #         solution[i] = best_pred
    #
    #     print("Building joined observation graph... ", end="", flush=True)
    #     t0 = time()
    #     joined_labels = np.zeros_like(observed_labels)
    #     for element, prediction in enumerate(solution):
    #         joined_labels[observed_labels==element]=prediction
    #     t1 = time()
    #     joined_patient = deepcopy(observed_patient)
    #     joined_patient.labelmaps["t2"].data = joined_labels
    #     observation_graph = liver_build_from_patient(joined_patient, vertex_mean, vertex_std, edge_mean, edge_std)
    #     print("Done. {:.4f}s + {:.4f}s".format(t1-t0,time()-t1))
    #
    #     display_volume(joined_labels,cmap=ListedColormap(class_colors))
