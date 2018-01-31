"""
SVM Segmentator of MRI liver patches with hard sample retraining
"""
import io_liver

import sys
from time import time
import matplotlib.pyplot as plt
import numpy as np
import random

from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
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
          data.append((x,y,z, data_window)) # Appending coordinates and window intensity
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
if len(sys.argv) < 7:
    print("Usage: svm_liver_hard_samples.py [window (ex: 553)] [pca components] [epochs] [convergence] [starting sample] [hard sample pct]")
_, window_parameter, pca_parameter, epochs_parameter, convergence_parameter, starting_sample_parameter, hard_pct_parameter = sys.argv[:7]
window = (int(window_parameter[0]), int(window_parameter[1]), int(window_parameter[2]))
n_components = int(pca_parameter)
epochs = int(epochs_parameter)
convergence_accuracy = float(convergence_parameter)
starting_sample_size = int(starting_sample_parameter)
hard_sample_pct = float(hard_pct_parameter)

print("Running SVM experiment for window {}, with {} components".format(window, pca_parameter))
print("Converge at {}{}, start with {}, hard sample percentage: {:.2f}%".format(epochs if epochs else convergence_accuracy, " epochs" if epochs else " accuracy", starting_sample_parameter,hard_sample_pct*100))

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

n_samples = len(data)

# Organizing the feature vector as 'x, y, z, {window intensities}'
X = np.array([np.concatenate(([x], [y], [z], window_intensities)) for x,y,z,window_intensities in data])
n_features = X.shape[1]

# the label to predict is foreground or background
y = np.array(target)
y = y.reshape((y.shape[0],))
target_names = ['0', '1']
n_classes = 2

###############################################################################
# Extracting initial train set with an equal class distribution

# shuffling dataset
shuffled_indexes = list(range(len(y)))
random.shuffle(shuffled_indexes)
#combined = list(zip(X, y))
#random.shuffle(combined)
#X[:], y[:] = zip(*combined)

# building starting trainset
X_train, y_train = [], []
fg_added, bg_added = 0, 0
for index in shuffled_indexes:
  if y[index] == 0:
    if bg_added < starting_sample_size:
      X_train.append(X[index])
      y_train.append(y[index])
      bg_added += 1
  if y[index] == 1:
    if fg_added < starting_sample_size:
      X_train.append(X[index])
      y_train.append(y[index])
      fg_added += 1
  if fg_added >= starting_sample_size and bg_added >= starting_sample_size:
    break

X_train, y_train, X_test, y_test = np.array(X_train), np.array(y_train), X, y
print("Initial set sizes: train - {} (fg: {}); test - {} (fg: {})".format(len(X_train), np.unique(y_train, return_counts=True)[1][1], len(X_test), np.unique(y_test, return_counts=True)[1][1]))

##############################################################################
# Compute a PCA (eigenfaces) on the full dataset (cheating but ok)

#n_components = 20

print("Extracting the top %d PCA components from %d elements... "
      % (n_components, X.shape[0]), end="", flush=True)
t0 = time()
pca = PCA(n_components=n_components, svd_solver='randomized',
          whiten=True).fit(X)
print("done in %0.3fs" % (time() - t0))

###############################################################################
# Run hard sample algorithm until convergence
current_epoch = 0
current_accuracy = 0

clf = None
# Training loop
while(current_epoch < epochs and current_accuracy < convergence_accuracy):
  print("On epoch {}. Training set has size {} (fg: {}), testing set has size {} (fg: {})".format(current_epoch, len(X_train), np.unique(y_train, return_counts=True)[1][1], len(X_test), np.unique(y_test, return_counts=True)[1][1]))
  ###############################################################################
  # Compute a PCA (eigenfaces) on the face dataset (treated as unlabeled
  # dataset): unsupervised feature extraction / dimensionality reduction

  #n_components = 20

  #print("Extracting the top %d PCA components from %d elements... "
  #      % (n_components, X_train.shape[0]), end="", flush=True)
  #t0 = time()
  #pca = PCA(n_components=n_components, svd_solver='randomized',
  #          whiten=True).fit(X_train)
  #print("done in %0.3fs" % (time() - t0))

  #eigenfaces = pca.components_.reshape((n_components, n_features))

  print("Projecting the input data on the eigenfaces orthonormal basis... ", end="", flush=True)
  t0 = time()
  X_train_pca = pca.transform(X_train)
  X_test_pca = pca.transform(X_test)
  print("done in %0.3fs" % (time() - t0))

  ###############################################################################
  # Train a SVM classification model

  print("Fitting the classifier to the training set... ", end="", flush=True)
  t0 = time()
  #param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
  #              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
  param_grid = {'C': [1e3],
                'gamma': [0.1], }
  clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
  clf = clf.fit(X_train_pca, y_train)
  print("done in %0.3fs" % (time() - t0))
  #print("Best estimator found by grid search:")
  #print(clf.best_estimator_)

  ###############################################################################
  # Testing the model on the test set
  print("Predicting bg/fg patches in the test set... ", end="", flush=True)
  t0 = time()
  y_pred = clf.predict(X_test_pca)
  print("done in %0.3fs" % (time() - t0))

  #current_accuracy = accuracy_score(y_test, y_pred)
  # Recall: TP/P
  #current_accuracy = sum([1 for pred, true in zip(y_pred, y_test) if (true == 1) and (true == pred)])/(np.unique(y_test, return_counts=True)[1][1])
  # Precision: TP/(TP+FP)
  current_accuracy = sum([1 for pred, true in zip(y_pred, y_test) if (true == 1) and (true == pred)])/sum([1 for pred, true in zip(y_pred, y_test) if (pred == 1)])
  print("* fg accuracy: {:.3f}".format(current_accuracy))
  print(classification_report(y_test, y_pred, target_names=target_names))
  print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))

  ###############################################################################
  # Selecting hard samples
  hard_samples_indexes = [idx for idx, targets in enumerate(zip(y_test, y_pred)) if targets[0] != targets[1]]
  hard_samples_chosen = random.sample(hard_samples_indexes, int(hard_sample_pct*len(hard_samples_indexes)))

  print("Picked {} hard samples out of {}".format(len(hard_samples_chosen), len(hard_samples_indexes)))

  # Adding hard samples to train
  X_hard_samples = X_test[hard_samples_chosen]
  y_hard_samples = y_test[hard_samples_chosen]

  X_train = np.concatenate((X_train, X_hard_samples))
  y_train = np.concatenate((y_train, y_hard_samples))

  # updating epoch
  print("End of epoch {}\n------------------------------------------\n".format(current_epoch))
  current_epoch += 1

###############################################################################
# Quantitative evaluation of the model quality on the full set

print("Full set evaluation. Done in {} epochs, final mean accuracy {:.3f}".format(current_epoch, current_accuracy))

#X = np.concatenate((X_train, X_test))
#y = np.concatenate((y_train, y_test))
X = np.array(X)
y = np.array(y)

#print("Extracting the top %d PCA components from %d elements... "
#      % (n_components, X.shape[0]), end="", flush=True)
#t0 = time()
#pca = PCA(n_components=n_components, svd_solver='randomized',
#          whiten=True).fit(X)
#print("done in %0.3fs" % (time() - t0))

#eigenfaces = pca.components_.reshape((n_components, n_features))

print("Projecting the input data on the eigenfaces orthonormal basis... ", end="", flush=True)
t0 = time()
X_pca = pca.transform(X)
print("done in %0.3fs" % (time() - t0))

print("Predicting bg/fg patches in the full set")
t0 = time()
y_pred = clf.predict(X_pca)
print("done in %0.3fs" % (time() - t0))

print(classification_report(y, y_pred, target_names=target_names))
print(confusion_matrix(y, y_pred, labels=range(n_classes)))


# Store model
from sklearn.externals import joblib
joblib.dump(clf, 'models/split_pca{}_window{}.pkl'.format(pca_parameter, window_parameter)) 