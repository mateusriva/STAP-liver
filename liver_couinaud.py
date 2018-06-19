"""Liver segmentation module for the SRG.

This module contains specific configurations
for the SRG, in order to make it segment liver
substructures.

Authors:
 * Mateus Riva (mriva@ime.usp.br)
"""

import numpy as np
from patient import Patient
from srg import SRG
from scipy.ndimage.measurements import center_of_mass as measure_center_of_mass

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

def liver_build_from_patient(patient):
    """Assembles a liver SRG from an annotated patient.

    Builds a fully-connected SRG from an annotated patient,
    for liver subsegmentation, utilizing the following attributes:
    * Structural: real centroid, absolute intensity, real volume, #semi-major axis
    * Relational: real centroid vector, relative intensity, relative volume, #relative orientation

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
    vertices = (vertices - vertices.mean(axis=0)) / vertices.std(axis=0)

    # Compute relational attributes
    positions = np.repeat(centroids, centroids.shape[0],axis=0) - np.vstack([centroids]*centroids.shape[0])
    contrasts = np.repeat(intensities, intensities.shape[0],axis=0) / np.vstack([intensities]*intensities.shape[0])
    ratios = np.repeat(sizes, sizes.shape[0],axis=0) / np.vstack([sizes]*sizes.shape[0])
    # Assemble relational attributes as the edges matrix
    edges = np.concatenate([positions, contrasts, ratios],axis=-1)
    # Normalizing to normal~(0,1)
    edges = (edges - edges.mean(axis=0)) / edges.std(axis=0)

    # Initializing the SRG
    return SRG(vertices, edges, ["centroid", "intensity", "size"], ["position", "contrast", "ratio"])

if __name__ == '__main__':
    from time import time

    print("Loading a single patient... ", end="", flush=True)
    t0 = time()
    model_patient = Patient.build_from_folder("data/4")
    print("Done. {:.4f}s".format(time()-t0))

    # We will be cutting the patient's volume and labelmap, just for speeding up the test
    model_patient.volumes["t2"].data = model_patient.volumes["t2"].data[:,:,20:]
    model_patient.labelmaps["t2"].data = model_patient.labelmaps["t2"].data[:,:,20:]

    # Splitting the background into 3 labels
    model_patient.labelmaps["t2"].data += 2 # Adding space for the extra labels at the start
    model_patient.labelmaps["t2"].data[np.logical_and(model_patient.volumes["t2"].data < 10, model_patient.labelmaps["t2"].data == 2)] = 0 # posterior background is 0
    model_patient.labelmaps["t2"].data[model_patient.labelmaps["t2"].data.shape[1]//2:,:,:][model_patient.labelmaps["t2"].data[model_patient.labelmaps["t2"].data.shape[1]//2:,:,:] == 0] = 1 # anterior background is 1
    model_patient.labelmaps["t2"].header["num_labels"] += 2

    print("Building model graph... ", end="", flush=True)
    t0 = time()
    model_graph = liver_build_from_patient(model_patient)
    print("Done. {:.4f}s".format(time()-t0))


    print(model_graph)
