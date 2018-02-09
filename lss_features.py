"""
This script extracts and organizes useful features from segmented
liver 3D patches.
Author:
 * Mateus Riva (mriva@ime.usp.br)
"""

import numpy as np

import lss_util
printv = lss_util.printv
printvv = lss_util.printvv

def extract_features(patch_dataset, features_string):
    """
    Extracts specified features from a patch, and adds them to
    the patch as a feature numpy array.

    Parameters
    ----------
    * patch_dataset: list
        List of patches, as returned by `lss_data.patch_dataset`.
    * features_string: string
        Feature code string (for more information, see README [TODO])

    Returns
    -------
    * dataset: list
        List of patches with extracted feature vectors
    * feature_count: int
        Total of extracted features
    """
    # Extracting and appending features to each individual patch
    for patch in patch_dataset:
        patch['features'], feature_count = extract_features_from_patch(patch, features_string)

    return patch_dataset, feature_count


def extract_features_from_patch(patch, features_string):
    """
    Extracts specified features from a patch, and adds them to
    the patch as a feature numpy array.

    Parameters
    ----------
    * patch: dict {data, target}
        Dict containing patch information, as returned by `lss_data.patch_dataset`.
    * features_string: string
        Feature code string (for more information, see README [TODO])

    Returns
    -------
    * features: array
        Array of extracted features
    #* patch: dict {data, features, target}
    #    Dict containing patch information, plus a feature vector
    """

    # initializing feature vector
    #patch['features'] = []
    features = []

    for feature_identifier in features_string.split(','):
        # Extracting each specific identified feature
        current_features = get_feature_function(feature_identifier)(patch)
        features.append(current_features)

    # Converting features to a flat array
    features = np.concatenate(features).flatten()
    return features, len(features)


def get_feature_function(feature_identifier):
    """
    Returns the proper feature extraction function for a given
    identifier. Basically a big switch.
    """
    # Coordinates feature
    if feature_identifier in ['coord', 'coords', 'coordinate', 'coordinates']:
        return coordinates
    # Intensities feature
    if feature_identifier in ['intens', 'intensity', 'intensities']:
        return intensities


def coordinates(patch):
    """
    Get coordinates of patch. TODO: center or uppertopleft coords?
    """
    return patch['data']['coords']

def intensities(patch):
    """
    Get intensities of patch as a flat list.
    """
    return patch['data']['intensities'].flatten()