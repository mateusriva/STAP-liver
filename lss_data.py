"""
This script reads, writes, and manipulates raw data relating to
the supervised liver dataset.
Author:
 * Mateus Riva (mriva@ime.usp.br)
"""

import os
import random
import numpy as np
import dicom
import nrrd
from natsort import natsorted
from scipy import stats

import lss_util
printv = lss_util.printv
printvv = lss_util.printvv

def load_folder(folder):
    """
    Loads and returns patient folders in the target folder. Walks to subfolders.
    A folder is considered a patient folder if it contains both of the following:
     * a folder named 'DixonZ', where Z = 1,2,3,4
     * a file ending in ".nrrd"
    """
    # Variable for containing each patient data tuple
    dataset = []

    # Walking all subfolders (including the top) and attempting to find patient folders
    for dirpath, dirnames, filenames in os.walk(folder):
        if any("Dixon" in dirname for dirname in dirnames) and any(filename.endswith(".nrrd") for filename in filenames):
            dataset.append((load_patient_folder(dirpath)))

    return dataset

def load_patient_folder(folder):
    """
    Loads and returns DICOM files and NRRD segmentations in the target folder, if available.
    Files must be in the following format:
    * mDixon captures:
        - <folder>/Dixon1/*.dcm (water); <folder>/Dixon2/*.dcm (inphase); <folder>/Dixon3/*.dcm (outphase); <folder>/Dixon4/*.dcm (fat)
    * T2 captures:
        - <folder>/T2/*.dcm
    * Segmentations:
        - <folder>/*.nrrd. Must contain "liver" or "spine" in the name; must contain "T2" or "mdixon" in the name.
    """
    
    # Loading dicom volumes
    volumes = {}
    for path, dirs, files in os.walk(folder):
        for dixon_folder in ["Dixon1", "Dixon2", "Dixon3", "Dixon4"]:
            if dixon_folder in dirs:
                dicom_data = load_dicom_folder(os.path.join(path, dixon_folder))
                volumes[dixon_folder] = dicom_data
        if "T2" in dirs:
            dicom_data = load_dicom_folder(os.path.join(path, "T2"))
            volumes["T2"] = dicom_data
    printvv("Loaded {} volumes.".format(len(volumes)))

    # Loading nrrd masks
    masks = {}
    for path, dirs, files in os.walk(folder):
        for file in files: 
            for structure in ["liver", "spine", "skin"]:
                for sequence in ["dixon", "t2"]:
                    if structure in file.lower() and sequence in file.lower() and ".nrrd" in file.lower():
                        mask_data = load_nrrd(os.path.join(path, file))
                        masks[sequence] = {}
                        masks[sequence][structure] = mask_data
    printvv("Loaded {} masks.".format(len(masks)))

    return volumes, masks

def load_nrrd(file):
    """
    Loads and returns a NRRD file.
    """
    printvv("Loading nrrd file {}".format(file))
    data, header = nrrd.read(file)

    # The transpose is used on raw, unaligned nrrds
    #return {"data": data.transpose((1,0,2)), "header" : header}
    return {"data": data, "header" : header}

def load_dicom_folder(folder):
    """
    Loads and returns a set of DICOM files from the target folder.
    """
    printvv("Loading dicom folder {}".format(folder))

    # Iterating over all DICOM files and storing the filenames:
    dicom_filenames = []
    for path, dirs, files in os.walk(folder):
        for file in natsorted(files):
            if ".dcm" in file.lower():
                dicom_filenames.append(os.path.join(path, file))
    # Read first file for metadata
    dicom_reference_file = dicom.read_file(dicom_filenames[0])
    # Load dimensions based on the number of rows, columns, and slices (along the Z axis)
    dicom_dimensions = (int(dicom_reference_file.Rows), int(dicom_reference_file.Columns), len(dicom_filenames))
    # Load spacing values (in mm)
    dicom_pixel_spacings = (float(dicom_reference_file.PixelSpacing[0]), float(dicom_reference_file.PixelSpacing[1]), float(dicom_reference_file.SliceThickness))

    # Generate empty numpy array for containing the data
    dicom_data = np.zeros(dicom_dimensions, dtype=dicom_reference_file.pixel_array.dtype)

    # loop through all the DICOM files
    for slice_count, dicom_filename in enumerate(dicom_filenames):
        # read the file
        dicom_file = dicom.read_file(dicom_filename)
        # store the raw image data
        dicom_data[:, :, slice_count] = dicom_file.pixel_array

    return {"dimensions":dicom_dimensions, "spacings":dicom_pixel_spacings, "data":dicom_data}


def patch_dataset(dataset, window, step=None):
    """
    Converts a loaded dataset into several pairs of 3D patches and corresponding targets
    A `step` of None is equal to the window (no overlap, no skip)

    Parameters
    ----------
     * dataset: list of tuples (volumes, masks)
         Dataset containing a list of volumes and masks per patient, as returned
         by load_folder
     * window: tuple (int, int, int)
         Window size, such as (5,5,3) for a 5x5x3 window
     * step: tuple (int, int, int)
         Step size, similar to window size

    Returns
    -------
      * patches: list of dicts {data, target}
          List of segmented 3D patches and corresponding targets
      * patch_stats: dict
          Statistics on the extracted patches
    """
    if step is None:
        step = window

    # Creating empty data vectors
    patches = []

    # Statistics (TODO: VARIABLE NUMBER OF CLASSES, or keep hardcoded?)
    patches_count = [0,0]

    # Extracting windows and targets
    for volumes, masks in dataset:

        # Sliding window
        data_shape = volumes["Dixon1"]["data"].shape
        for x in range(0,data_shape[0]-step[0],step[0]):
            for y in range(0,data_shape[1]-step[1],step[1]):
                for z in range(0,data_shape[2]-step[2],step[2]):
                    # Concatenating all dixon sequences into a 4D window
                    data_window = np.stack([volumes["Dixon1"]["data"][x:x+window[0],y:y+window[1],z:z+window[2]],volumes["Dixon2"]["data"][x:x+window[0],y:y+window[1],z:z+window[2]],volumes["Dixon3"]["data"][x:x+window[0],y:y+window[1],z:z+window[2]],volumes["Dixon4"]["data"][x:x+window[0],y:y+window[1],z:z+window[2]]], axis=-1)
                    #data_window = data_window.flatten()

                    # Ignoring empty patches
                    if not data_window.any():
                        continue

                    # Getting corresponding targets for this window and assessing mode
                    target_window = masks["dixon"]["liver"]["data"][x:x+window[0],y:y+window[1],z:z+window[2]]
                    target = int(stats.mode(target_window, axis=None)[0][0])
                    data= {"coords": np.array([x,y,z]), "intensities": data_window} # Extracted data: coordinates and window intensity
                    patches.append({'data': data, 'target': target})

                    # Counting statistics
                    patches_count[target] = patches_count[target] + 1

    patch_stats = {"total": sum(patches_count), "per_class": patches_count}
    return patches, patch_stats

def assemble_initial_train_set(X, y, initial_train_size):
    """
    Assembles a class-balanced initial training set from the full set.
    """
    feature_count = X.shape[1]
    # Initializing train set as numpy arrays
    X_train, y_train = np.empty((initial_train_size*2, feature_count)), np.empty((initial_train_size*2))

    # Shuffling dataset for sampling
    shuffled_indexes = list(range(len(y)))
    random.shuffle(shuffled_indexes)

    # Building balanced starting trainset (TODO: update this for multiclass)
    fg_added, bg_added = 0, 0
    for index in shuffled_indexes:
      if y[index] == 0:
        if bg_added < initial_train_size:
          X_train[bg_added] = X[index]
          y_train[bg_added] = y[index]
          bg_added += 1
      if y[index] == 1:
        if fg_added < initial_train_size:
          X_train[fg_added] = X[index]
          y_train[fg_added] = y[index]
          fg_added += 1
      if fg_added >= initial_train_size and bg_added >= initial_train_size:
        break

    return X_train, y_train