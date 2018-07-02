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

def compute_centroids(volume, labelmap):
    """Computes centroids for each label in a volume.

    Returns
    -------
    centroids : `2darray`
        Array with `num_labels` lines and 3 columns. Each line contains
        the normalized, real x, y, and z of each label.
    """
    # get sorted labels and center-of-mass for each label
    labels = np.unique(labelmap.data)
    centroids = measure_center_of_mass(np.ones_like(labelmap.data), labels=labelmap.data, index=labels)
    centroids = np.array(centroids)

    # multiply voxel value by voxel size to get real centroid
    centroids *= volume.header["spacings"]

    # normalizing to normal~(0,1)
    #centroids = (centroids - centroids.mean(axis=0)) / centroids.std(axis=0)

    return centroids

def compute_intensities(volume, labelmap):
    """Computes mean intensities for each label in a volume.

    TODO: some volume normalization? Gradient, maybe? Or mean norm of voxels?

    Returns
    -------
    intensities : `2darray`
        Array with `num_labels` lines and 1 column. Each line contains
        the absolute mean intensity of each label.
    """
    # get sorted labels and indexes for each label
    labels, indexes = np.unique(labelmap.data, return_inverse=True)
    # initializing intensities array
    intensities = np.empty((len(labels),1))

    # building mean intensity attribute list
    for label in labels:
        intensities[label] = np.mean(volume.data.flatten()[indexes==label])

    # normalizing to normal~(0,1)
    #intensities = (intensities - intensities.mean(axis=0)) / intensities.std(axis=0)

    return intensities

def compute_sizes(volume, labelmap):
    """Computes the size (volume) of each label in a volume.

    Returns
    -------
    sizes : `2darray`
        Array with `num_labels` lines and 1 column. Each line contains
        the absolute size of each label.
    """
    # get sorted labels and voxel count per label
    labels,voxel_count_per_labels = np.unique(labelmap.data, return_counts=True)
    # multiplying by the real size of the voxel
    voxel_count_per_labels = voxel_count_per_labels * np.prod(volume.header["spacings"])
    voxel_count_per_labels = voxel_count_per_labels.reshape((voxel_count_per_labels.shape[0],1))

    # normalizing to normal~(0,1)
    #voxel_count_per_labels = (voxel_count_per_labels - voxel_count_per_labels.mean(axis=0)) / voxel_count_per_labels.std(axis=0)

    return voxel_count_per_labels

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

    # Self-matching for calibration
    print("Calibrating model... ", end="", flush=True)
    t0 = time()
    other_graph = deepcopy(model_graph)
    solution = np.empty(other_graph.vertices.shape[0])
    # Getting closest model vertex, for each vertex
    for i, vertex in enumerate(other_graph.vertices):
        vertex_matrix = np.vstack([vertex]*model_graph.vertices.shape[0])
        distances = np.linalg.norm((model_graph.vertices[:,:-1]-vertex_matrix[:,:-1]), axis=-1)#np.sum(abs(model_graph.vertices-vertex_matrix),axis=-1)
        solution[i] = np.argmin(distances)
    errors = np.sum(solution != np.arange(model_graph.vertices.shape[0]))
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
    observed_patient.labelmaps["t2"] = observed_labels
    super_observation_graph = liver_build_from_patient(observed_patient, vertex_mean, vertex_std, edge_mean, edge_std)
    print("Done. {:.4f}s".format(time()-t0))

    print(represent_liver_srg(model_graph,True))

    # Attempting various weights
    # costs, accuracies = [], []
    # weights_list = product([0,1,5,9], repeat=4)
    # for weights in weights_list:
    #     print("Assembling initial prediction with weights {}... ".format(weights), end="", flush=True)
    #     t0 = time()
    #     solution = np.empty(super_observation_graph.vertices.shape[0])
    #     # Getting closest model vertex, for each vertex
    #     for i, vertex in enumerate(super_observation_graph.vertices):
    #         vertex_matrix = np.vstack([vertex]*model_graph.vertices.shape[0])
    #         distances = np.linalg.norm((weights*(model_graph.vertices[:,:-1]-vertex_matrix[:,:-1])/np.sum(weights)), axis=-1)#np.sum(abs(model_graph.vertices-vertex_matrix),axis=-1)
    #         solution[i] = np.argmin(distances)
    #     print("Done. {:.4f}s".format(time()-t0))
    #
    #     # Displaying a prediction
    #     #display_solution(observed_labels, solution, title="Weights: {}".format(weights),cmap=ListedColormap(np.array(class_colors)[np.unique(solution).astype(int)]))
    #     # Computing total prediction cost
    #     cost = np.sum([np.linalg.norm(weights*(model_graph.vertices[int(solution[x]),:-1]-super_observation_graph.vertices[x,:-1]), axis=-1) for x in range(len(super_observation_graph.vertices))])
    #     costs.append(cost)
    #     predicted_labelmap = np.zeros_like(observed_labels)
    #     for element, prediction in enumerate(solution):
    #         predicted_labelmap[observed_labels==element]=prediction
    #     accuracy = np.sum([predicted_labelmap==model_patient.labelmaps['t2'].data]) / predicted_labelmap.size
    #     accuracies.append(accuracy)
    #
    #     print("Accuracy: {:.2f}%, Cost: {:.2f}".format(accuracy*100,cost))
    #
    #     del vertex_matrix, distances, predicted_labelmap, cost, accuracy
    #
    # for cost,accuracy,weights in zip(*sorted(zip(costs,accuracies,weights_list))):
    #     print("W {} : Accuracy: {}%, Cost: {:.3f}".format(weights,accuracy,cost))

    weights = [1,5,5,9]
    print("Assembling initial prediction with weights {}... ".format(weights), end="", flush=True)
    t0 = time()
    solution = np.empty(super_observation_graph.vertices.shape[0])
    # Getting closest model vertex, for each vertex
    for i, vertex in enumerate(super_observation_graph.vertices):
        vertex_matrix = np.vstack([vertex]*model_graph.vertices.shape[0])
        distances = np.linalg.norm((weights*(model_graph.vertices[:,:-1]-vertex_matrix[:,:-1])/np.sum(weights)), axis=-1)#np.sum(abs(model_graph.vertices-vertex_matrix),axis=-1)
        solution[i] = np.argmin(distances)
    print("Done. {:.4f}s".format(time()-t0))

    # Joining regions into an observation graph
    print("Building joined observation graph... ", end="", flush=True)
    t0 = time()
    joined_labels = np.zeros_like(observed_labels)
    for element, prediction in enumerate(solution):
        joined_labels[observed_labels==element]=prediction
    joined_patient = deepcopy(observed_patient)
    joined_patient.labelmaps["t2"].data = joined_labels
    observation_graph = liver_build_from_patient(joined_patient, vertex_mean, vertex_std, edge_mean, edge_std)
    print("Done. {:.4f}s".format(time()-t0))

    print(represent_liver_srg(observation_graph,True))

    # Greedy improvement of joined observation graph (by changing the super observation graph)
