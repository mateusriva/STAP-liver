"""Module for displaying medical data in a user-readable fashion.

This module basically extends `matplotlib.pyplot`'s functions for
image display in a way that makes it easy to visualize MRI slices
and corresponding labelmaps.
NOTE: these are specific to the Iron-ICr project. Adapt as needed.

Authors:
 * Mateus Riva (mriva@ime.usp.br)
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import data, color, io, img_as_float

from lic_patient import Patient

label_color_map = {
    0: (0,0,0),         # Background: no label
    1: (0,0,1),         # Vena Cava: blue
    2: (0,1,1),         # Portal Vein: light blue
    3: (0.5,0,1),       # Left Hepatic Vein: purplish-blue
    4: (0,0.5,1),       # Middle Hepatic Vein: light-but-not-so-much blue
    5: (0,0,0.5),       # Right Hepatic Vein: dark blue
    6: (1,0,0),         # Segment I: red
    7: (1,0.5,0),       # Segment II: orange
    8: (0.5,0.5,0),     # Segment III: dark gold
    9: (0,1,0),         # Segment IVa: light green
    10: (0,0.5,0),      # Segment IVb: dark green
    11: (1,0.5,1),      # Segment V: light pink
    12: (0.5,0,0.5),    # Segment VI: purple
    13: (1,1,0),        # Segment VII: yellow
    14: (1,0,1),        # Segment VIII: pink
}
"""dict: mapping of voxel label to visualization color."""

def overlay_labeled_slice(volume_slice, labelmap_slice, label_opacity=1.0, window_wl=None):
    """Overlays a labelmap on a slice with given opacity.
    """
    assert len(volume_slice.shape) == 2, "volume slice is not 2D"
    assert len(labelmap_slice.shape) == 2, "label slice is not 2D"

    if window_wl is not None:
        width, level = window_wl
    else:
        width, level = np.max(volume_slice), 0

    # Normalizing image to window level
    display_slice = volume_slice.astype(float)
    display_slice = (display_slice-(level-(width/2)))/(width)
    display_slice[display_slice < 0.0] = 0.0
    display_slice[display_slice > 1.0] = 1.0

    # Building RGB image from gray-slice and converting to HSV
    display_slice = display_slice.repeat(3,1).reshape((display_slice.shape[0], display_slice.shape[1], 3))
    display_slice = color.rgb2hsv(display_slice)

    # building color slice from labelmap and converting to HSV
    color_slice = np.empty((labelmap_slice.shape[0], labelmap_slice.shape[1], 3), dtype=float)
    for x,row in enumerate(labelmap_slice):
        for y,pixel in enumerate(row):
            color_slice[x,y] = label_color_map[pixel]
    color_slice = color.rgb2hsv(color_slice)

    # Replacing hue and saturation of original image with that of color mask
    # Source: https://stackoverflow.com/a/9204506
    display_slice[..., 0] = color_slice[..., 0]
    display_slice[..., 1] = color_slice[..., 1] * label_opacity
    display_slice = color.hsv2rgb(display_slice)

    # Hack: if opacity is 1, simply replace values in the image
    # (otherwise, it will never achieve full opacity)
    if label_opacity >= 1.0:
        color_slice = color.hsv2rgb(color_slice)
        display_slice[labelmap_slice > 0] = color_slice[labelmap_slice > 0]

    return display_slice

if __name__ == '__main__':
    patient = Patient.build_from_folder("data/4")
    volume_slice, labelmap_slice = patient.volumes['t2'].data[:,:,30], patient.labelmaps['t2'].data[:,:,30]
    plt.imshow(overlay_labeled_slice(volume_slice,labelmap_slice, window_wl=(700,300), label_opacity=0.9))
    #plt.imshow(volume_slice,cmap='gray')
    plt.show()