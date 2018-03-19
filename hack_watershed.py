"""This module is a set of hacks and quick scripts for achieving best watershed. Delete for release.
"""

import numpy as np
from skimage.morphology import watershed, local_minima
import scipy.ndimage as ndi
import matplotlib, matplotlib.pyplot as plt

from lic_patient import Patient

# importing patient and selecting volume
patient = Patient.build_from_folder("data/4")
volume = patient.volumes["t2"].data
# thresholding background noise
volume[volume < 10] = 0
# normalizing
volume = volume / np.max(volume)

# gaussian filtering with (5,5,3)
volume553 = ndi.gaussian_filter(volume, (5,5,3))
volume551 = ndi.gaussian_filter(volume, (5,5,1))
volume331 = ndi.gaussian_filter(volume, (3,3,1))

#showing gaussians
# plt.gray()
# plt.subplot(221)
# plt.title("Original")
# plt.imshow(volume[:,:,36])
# plt.subplot(222)
# plt.title("Gaussian 5,5,3")
# plt.imshow(volume553[:,:,36])
# plt.subplot(223)
# plt.title("Gaussian 5,5,1")
# plt.imshow(volume551[:,:,36])
# plt.subplot(224)
# plt.title("Gaussian 3,3,1")
# plt.imshow(volume331[:,:,36])
# plt.show()

# sobel vs morph gradient
morphgrad = ndi.morphology.morphological_gradient(volume331, (19,19,5))

# showing grads
# plt.subplot(221)
# plt.title("Sobel Gradient")
# plt.imshow(sobelgrad[:,:,36])
# plt.subplot(222)
# plt.title("Morph Gradient")
# plt.imshow(morphgrad[:,:,36], cmap="gray")
# plt.subplot(223)
# plt.title("Morph Gradient 5x5x3")
# plt.imshow(morphgrad553[:,:,36])
# plt.subplot(224)
# plt.title("Morph Gradient 3x3x1")
# plt.imshow(morphgrad331[:,:,36])
# plt.show()

# local minima per gradient
morphgrad_minima = local_minima(morphgrad, selem=np.ones((5,5,5)))

# showing minima markers
#plt.imshow(morphgrad_minima[:,:,36])
#plt.show()

# labeling minima
morphgrad_markers, morphgrad_total_markers = ndi.label(morphgrad_minima)

# performing watershed
morphgrad_watershed = watershed(morphgrad, morphgrad_markers, watershed_line=False)
morphgrad_lines = watershed(morphgrad, morphgrad_markers, watershed_line=True)

cmap = matplotlib.colors.ListedColormap ( np.random.rand ( 256,3))

# showing watersheds
plt.title("Morph Gradient 19,19,5 - {} regions".format(len(np.unique(morphgrad_watershed))))
plt.imshow(morphgrad_watershed[:,:,36],cmap=cmap)
plt.show()

# overlaying watershed lines
def overlay_line(image, ws):
    # gray to rgb
    overlayed = image.repeat(3,1).reshape((image.shape[0], image.shape[1], 3))
    # red lines
    overlayed[ws == 0] = [1,0,0]
    return overlayed

morphgrad_overlay = overlay_line(volume[:,:,36], morphgrad_lines[:,:,36])

# showing overlays
plt.title("Morph Gradient 19,19,5 - {} regions".format(len(np.unique(morphgrad_watershed))))
plt.imshow(morphgrad_overlay)
plt.show()