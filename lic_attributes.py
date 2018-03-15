"""Module for computing attributes from a given patient object.

Currently, three attributes are supported:
* volumetry (computed as total of voxels * volume of individual voxel)
* centroid 
* mean intensity

Three relationships, based on these attributes, are also easily computed:
* proportional volume
* vectorial distance
* contrast

Authors:
 * Mateus Riva (mriva@ime.usp.br)
"""

import numpy as np
from scipy.ndimage.measurements import center_of_mass as measure_center_of_mass

from lic_patient import Patient

def compute_volumetry(patient):
    """Function for computing per-class volumetry of a patient.

    This function takes a Patient object as an argument and,
    using its corresponding labelmap(s) [TODO: support multiple
    labelmaps], extracts volumetry information for each class.

    This volumetry information comes in three flavors, each an
    element of the corresponding label's dict:
     * 'voxel': volumetry given as number of voxels
     * 'real': volumetry given as real total in mm³
     * 'relative': real volumetry normalized by patient total volume
    
    Attributes
    ----------
    patient : :obj:`Patient`
        Patient object to extract volumetry from.

    Returns
    -------
    volumetry : `dict`
        dict of volumetry information, keyed by label
    """
    # Acquiring the first labelmap of the patient (TODO: multiple labelmap support)
    labelmap = patient.labelmaps['t2']
    volume = patient.volumes['t2']

    # get sorted labels and voxel count per label
    labels,voxel_count_per_labels = np.unique(labelmap.data, return_counts=True)

    # building volumetry attribute list
    volumetry = {}
    for label, voxel_count in zip(labels, voxel_count_per_labels):
        volumetry[label] = {"voxel": voxel_count,
                            "real": voxel_count*np.prod(volume.header["spacings"]),
                            "relative": voxel_count*np.prod(volume.header["spacings"])/patient.total_volume["real"]}

    return volumetry


def compute_centroids(patient):
    """Function for computing per-class centroid of a patient.

    This function takes a Patient object as an argument and,
    using its corresponding labelmap(s) [TODO: support multiple
    labelmaps], extracts centroid information for each class.

    This centroid information comes in three flavors, each an
    element of the corresponding label's dict:
     * 'voxel': centroid given as index of voxel
     * 'real': centroid given as real position in mm
     * 'relative': voxel centroid normalized in the volume
    
    Attributes
    ----------
    patient : :obj:`Patient`
        Patient object to extract centroid from.

    Returns
    -------
    centroids : `dict`
        dict of centroid information, keyed by label
    """
    # Acquiring the first labelmap of the patient (TODO: multiple labelmap support)
    labelmap = patient.labelmaps['t2']
    volume = patient.volumes['t2']

    # get sorted labels and center-of-mass for each label
    labels = np.unique(labelmap.data)
    centers_of_mass = measure_center_of_mass(np.ones_like(labelmap.data), labels=labelmap.data, index=labels)

    # building centroid attribute list
    centroids = {}
    for label, centroid in zip(labels, centers_of_mass):
        centroids[label] = {"voxel": list(centroid),
                            "real": [(centroid[i]*volume.header["spacings"][i])+volume.header["initial_position"][i] for i in range(len(centroid))],
                            "relative": [centroid[i]/volume.data.shape[i] for i in range(len(centroid))]}

    return centroids


def compute_mean_intensities(patient):
    """Function for computing per-class mean intensity of a patient.

    This function takes a Patient object as an argument and,
    using its corresponding labelmap(s) and volume(s) [TODO: support
    multiple sequences], extracts mean intensity information for
    each class.

    This mean intensity information comes in two flavors, each an
    element of the corresponding label's dict:
     * 'real': mean intensity given as average intensity of radiometry
     * 'relative': real mean intensity normalized by the min/max of the volume
    
    Attributes
    ----------
    patient : :obj:`Patient`
        Patient object to extract mean intensity from.

    Returns
    -------
    mean_intensities : `dict`
        dict of mean intensity information, keyed by label
    """
    # Acquiring the first labelmap of the patient (TODO: multiple labelmap support)
    labelmap = patient.labelmaps['t2']
    volume = patient.volumes['t2']

    # get sorted labels and indexes for each label
    labels, indexes = np.unique(labelmap.data, return_inverse=True)

    # building mean intensity attribute list
    mean_intensities = {}
    for label in labels:
        mean_intensities[label] = {"real": np.mean(volume.data.flatten()[indexes==label])}
        mean_intensities[label]["relative"] = (mean_intensities[label]["real"]-np.min(volume.data))/(np.max(volume.data)-np.min(volume.data))

    return mean_intensities



if __name__ == '__main__':
    """Debug function. Delete me for release."""
    patient = Patient.build_from_folder("data/4")

    # Testing volumetry
    print("Total volumetry:", patient.total_volume)
    print("Patient volumetry:\n----------------")
    volumetry = compute_volumetry(patient)
    for key in volumetry:
        print("{}: {}".format(key, volumetry[key]))

    # Testing centroid
    print("Patient centroids:\n----------------")
    centroids = compute_centroids(patient)
    for key in centroids:
        print("{}: {}".format(key, centroids[key]))

    # Testing mean_intensity
    print("Patient min/max intensities:",np.min(patient.volumes['t2'].data), np.max(patient.volumes['t2'].data))
    print("Patient mean_intensities:\n----------------")
    mean_intensities = compute_mean_intensities(patient)
    for key in mean_intensities:
        print("{}: {}".format(key, mean_intensities[key]))
