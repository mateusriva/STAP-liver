import os

import numpy as np
#import matplotlib.pyplot as plt
from natsort import natsorted

import nrrd
import dicom

# debug print messages
debug = False
def dprint(string):
    if debug:
        print("#",string)

# hack: align segmentation data
def align_segmentation(volume, segmentation, name):
    dprint("Aligning {}".format(name))

    alignments = {"Nl 1": [71,24,28],
                "Nl 2": [67,28,11],
                "Nl 3": [75,17,12],
                "Caso 1": [48,25,25],
                "Caso 2": [64,32,12],
                "Caso 3": [81,51,13]}

    aligned_segmentation = np.zeros(volume.shape)
    alignment = alignments[name]
    aligned_segmentation[alignment[0]:alignment[0]+segmentation.shape[0],alignment[1]:alignment[1]+segmentation.shape[1],alignment[2]:alignment[2]+segmentation.shape[2]] = segmentation

    return aligned_segmentation

def load_data(folder):
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

    dprint("Loaded {} volumes.".format(len(volumes)))

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

    return volumes, masks

def load_nrrd(file):
    """
    Loads and returns a NRRD file.
    """
    dprint("Loading NRRD file {}".format(file))

    data, header = nrrd.read(file)
    return {"data": data.transpose((1,0,2)), "header" : header}

#['108.47787472488653', '22.404827472767614', '-8.3051706959686946']

def load_dicom_folder(folder):
    """
    Loads and returns a set of DICOM files from the target folder.
    """
    dprint("Loading DICOM folder {}".format(folder))

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



# Execution and tests
# debug = True
# volumes, masks = load_data("data/Nl 3")

# masks['dixon']['liver']['data'] = align_segmentation(volumes['Dixon1']['data'], masks['dixon']['liver']['data'], "Nl 3")
# print(masks['dixon']['liver']['data'].shape)

# for slice in range(10, volumes['Dixon1']['data'].shape[-1]):
#     current_slice_image = volumes['Dixon1']['data'][:,:,slice]
#     current_slice_mask = masks['dixon']['liver']['data'][:,:,slice] + 0.5

#     plt.imshow(current_slice_image*current_slice_mask, cmap='gray')
#     plt.show()

#plt.subplot(121)
#plt.imshow(volumes['Dixon1']['data'][:,:,40], cmap='gray')
#plt.subplot(122)
#plt.imshow(masks['dixon']['liver']['data'][:,:,30], cmap='gray')
#plt.show()