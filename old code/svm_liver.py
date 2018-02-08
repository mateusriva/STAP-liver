"""
SVM Segmentator of MRI liver patches
"""
import io_liver

import sys
from time import time
import matplotlib.pyplot as plt
import numpy as np

from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC



def discretize_dataset(dataset, window, step=None):
  """
  Converts a loaded dataset into several pairs of window->targets
  A `step` of None is equal to the window (no overlap, no skip)
  """
  if step is None:
    step = window

  # Creating empty data vectors
  data, targets = [], []

  # Extracting windows and targets
  for pack, patient in dataset:
    volumes,masks = pack

    # Aligning mask
    masks['dixon']['liver']['data'] = io_liver.align_segmentation(volumes['Dixon1']['data'], masks['dixon']['liver']['data'], patient)

    # Sliding window
    data_shape = volumes["Dixon1"]["data"].shape
    for x in range(0,data_shape[0]-step[0],step[0]):
      for y in range(0,data_shape[1]-step[1],step[1]):
        for z in range(0,data_shape[2]-step[2],step[2]):
          # Concatenating all dixon sequences into a 4D window
          data_window = np.stack([volumes["Dixon1"]["data"][x:x+window[0],y:y+window[1],z:z+window[2]],volumes["Dixon2"]["data"][x:x+window[0],y:y+window[1],z:z+window[2]],volumes["Dixon3"]["data"][x:x+window[0],y:y+window[1],z:z+window[2]],volumes["Dixon4"]["data"][x:x+window[0],y:y+window[1],z:z+window[2]]], axis=-1)
          data_window = data_window.flatten()

          # Ignoring empty patches
          if not data_window.any():
            continue

          # Getting corresponding targets for this window and assessing mode
          target_window = masks["dixon"]["liver"]["data"][x:x+window[0],y:y+window[1],z:z+window[2]]
          target_item = stats.mode(target_window, axis=None)[0]
          data.append(data_window)
          targets.append(target_item)

  return data, targets

def dataset_stats(data, targets):
  """ Print some stats on the dataset """
  print("Total of patches: {}".format(len(data)))
  unique, count = np.unique(targets, return_counts=True)
  totals = dict(zip(unique, count))
  print("Non-target patches: {} ({:.2f}%)".format(totals[0], (totals[0]*100/len(data))))
  print("    Target patches: {} ({:.2f}%)".format(totals[1], (totals[1]*100/len(data))))

###############################################################################
# Reading command-line arguments
if len(sys.argv) < 3:
    print("Usage: svm_liver.py [window (ex: 553)] [pca components]")
_, window_parameter, pca_parameter = sys.argv[:3]
window = (int(window_parameter[0]), int(window_parameter[1]), int(window_parameter[2]))
n_components = int(pca_parameter)

print("Running SVM experiment for window {}, with {} components".format(window, pca_parameter))

###############################################################################
# Load dataset from io.py

io_liver.debug = True

t0 = time()
#dataset = [(io_liver.load_data("data/{}".format(patient)), patient) for patient in ["Caso 2"]]
dataset = [(io_liver.load_data("data/{}".format(patient)), patient) for patient in ["Caso 1","Caso 2","Caso 3","Nl 1","Nl 2","Nl 3"]]

# Split dataset into windows and flatten it
data, target = discretize_dataset(dataset, window=window)
print(dataset_stats(data, target))
print("Loaded dataset in %0.3fs" % (time() - t0))

# introspect the images arrays to find the shapes (for plotting)
#n_samples, h, w = lfw_people.images.shape
n_samples = len(data)

# for machine learning we use the 2 data directly (as relative pixel
# positions info is ignored by this model)
X = np.array(data)
n_features = X.shape[1]

# the label to predict is the id of the person
y = np.array(target)
y = y.reshape((y.shape[0],))
target_names = ['0', '1']
n_classes = 2

#print("Total dataset size:")
#print("n_samples: %d" % n_samples)
#print("n_features: %d" % n_features)
#print("n_classes: %d" % n_classes)


###############################################################################
# Split into a training set and a test set using a stratified k fold

# split into a training and testing set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)

###############################################################################
# Compute a PCA (eigenfaces) on the face dataset (treated as unlabeled
# dataset): unsupervised feature extraction / dimensionality reduction

#n_components = 20

print("Extracting the top %d PCA components from %d elements"
      % (n_components, X_train.shape[0]))
t0 = time()
pca = PCA(n_components=n_components, svd_solver='randomized',
          whiten=True).fit(X_train)
print("done in %0.3fs" % (time() - t0))

#eigenfaces = pca.components_.reshape((n_components, n_features))

print("Projecting the input data on the eigenfaces orthonormal basis")
t0 = time()
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
print("done in %0.3fs" % (time() - t0))

###############################################################################
# Train a SVM classification model

print("Fitting the classifier to the training set")
t0 = time()
param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
#param_grid = {'C': [1e3],
#              'gamma': [0.1], }
clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
clf = clf.fit(X_train_pca, y_train)
print("done in %0.3fs" % (time() - t0))
print("Best estimator found by grid search:")
print(clf.best_estimator_)

# Store model
from sklearn.externals import joblib
joblib.dump(clf, 'models/split_pca{}_window{}.pkl'.format(pca_parameter, window_parameter)) 

###############################################################################
# Quantitative evaluation of the model quality on the test set

print("Predicting bg/fg patches in the test set")
t0 = time()
y_pred = clf.predict(X_test_pca)
print("done in %0.3fs" % (time() - t0))

print(classification_report(y_test, y_pred, target_names=target_names))
print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))
