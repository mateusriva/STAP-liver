"""
This script applies a saved SVM classifier model to a dicom folder, and saves it as .jpgs for qualitative analysis.
"""
import sys
from time import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn.externals import joblib
from sklearn.decomposition import PCA

import io_liver

if len(sys.argv) < 3:
    print("Usage: apply_model_to_patient <patient> <model>")
    exit(-1)

print("Applying model {} to patient {}".format(sys.argv[2], sys.argv[1]))

# Loading a patient
patient = sys.argv[1]
volumes, _ = io_liver.load_data("data/{}".format(patient))

# Loading the predictor model
model = sys.argv[2]
clf = joblib.load(model)

n_components = int(model.split("pca")[1].split("_")[0])
window_string = model.split("window")[1].split(".")[0]
window = (int(window_string[0]), int(window_string[1]), int(window_string[2]))
step = window

# Creating empty result cube
predicted_volume = np.zeros(volumes["Dixon1"]["data"].shape)

# Loading data over windows
data = []
# Sliding window
data_shape = volumes["Dixon1"]["data"].shape
for x in range(0,data_shape[0]-step[0],step[0]):
    for y in range(1,data_shape[1]-step[1],step[1]):
        for z in range(2,data_shape[2]-step[2],step[2]):
            # Concatenating all dixon sequences into a 4D window
            data_window = np.stack([volumes["Dixon1"]["data"][x:x+window[0],y:y+window[1],z:z+window[2]],volumes["Dixon2"]["data"][x:x+window[0],y:y+window[1],z:z+window[2]],volumes["Dixon3"]["data"][x:x+window[0],y:y+window[1],z:z+window[2]],volumes["Dixon4"]["data"][x:x+window[0],y:y+window[1],z:z+window[2]]], axis=-1)
            data_window = data_window.flatten()

            # Ignoring empty patches
            if not data_window.any():
                continue

            data.append(data_window)

X = np.array(data)


print("Extracting the top %d PCA components from %d elements"
      % (n_components, X.shape[0]))
t0 = time()
pca = PCA(n_components=n_components, svd_solver='randomized',
          whiten=True).fit(X)
print("done in %0.3fs" % (time() - t0))

#eigenfaces = pca.components_.reshape((n_components, n_features))

print("Projecting the input data on the eigenfaces orthonormal basis")
t0 = time()
X_pca = pca.transform(X)
print("done in %0.3fs" % (time() - t0))

# Iterating over windows and predicting
i = 0
# Sliding window
data_shape = volumes["Dixon1"]["data"].shape
for x in range(0,data_shape[0]-step[0],step[0]):
    for y in range(1,data_shape[1]-step[1],step[1]):
        for z in range(2,data_shape[2]-step[2],step[2]):
            # Concatenating all dixon sequences into a 4D window
            data_window = np.stack([volumes["Dixon1"]["data"][x:x+window[0],y:y+window[1],z:z+window[2]],volumes["Dixon2"]["data"][x:x+window[0],y:y+window[1],z:z+window[2]],volumes["Dixon3"]["data"][x:x+window[0],y:y+window[1],z:z+window[2]],volumes["Dixon4"]["data"][x:x+window[0],y:y+window[1],z:z+window[2]]], axis=-1)
            data_window = data_window.flatten()

            # Ignoring empty patches
            if not data_window.any():
                continue

            # Predicting
            prediction = clf.predict([X_pca[i]])[0]
            i += 1
            predicted_volume[x:x+window[0],y:y+window[1],z:z+window[2]] = prediction

for slice in range(10, volumes['Dixon1']['data'].shape[-1]):
    current_slice_image = volumes['Dixon1']['data'][:,:,slice]
    current_slice_mask = predicted_volume[:,:,slice] + 0.5

    plt.imshow(current_slice_image*current_slice_mask, cmap='gray')
    plt.title("{} prediction for {}, slice {}".format(patient, model, slice))
    plt.xticks()
    plt.yticks()
    plt.savefig("predict/{}/pca{}_window{}_slice{}.png".format(patient, n_components, window_string, slice))