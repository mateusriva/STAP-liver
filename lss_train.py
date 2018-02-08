"""
This script reads pre-segmented liver data, assembles a dataset,
extracts features, trains a SVM on the data and reports training
quality. The SVM model built is stored in the "./models" folder.

Author: 
 * Mateus Riva (mriva@ime.usp.br)
"""
import sys
from itertools import chain
from time import time
import numpy as np
import optparse

import lss_data, lss_features, lss_util
printv = lss_util.printv
printvv = lss_util.printvv


################################################################
# Reading and parsing command-line arguments
parser = optparse.OptionParser("usage: %prog [options] data_folder [data_folder2 ...]")

#parser.add_option("-d", "--data-folder", dest="data_folder",
#               default="error", type="string",
#               help="folder containing liver segmentation data (subfolders will be checked).")
parser.add_option("-w", "--window-size", dest="window_code", default="553",
               type="string", help="window size string: 'xyz'. Example: '553' uses a 5x5x3 window.")
parser.add_option("-f", "--features", dest="features_string", default="coordinates,intensity",
               type="string", help="features to be extracted from the data. Example: 'coordinates,intensity'. Don't use spaces, split by commas. Further information available at the README [TODO]")
parser.add_option("-c", "--components", dest="pca_components_total", default=20,
               type="int", help="number of PCA components to be extracted.")
parser.add_option("-e", "--epochs", dest="max_epochs", default=100,
               type="int", help="maximum number of training epochs.")
parser.add_option("-t", "--threshold", dest="convergence_threshold", default=0.95,
               type="float", help="threshold for convergence. Training stops if max epochs reached or if precision and recall of both classes exceed threshold.")
parser.add_option("-i", "--initial-size", dest="initial_train_size", default=1000,
               type="int", help="size of the initial (balanced) train set.")
parser.add_option("-l", "--learning-rate", dest="learning_rate", default=0.1,
               type="float", help="learning rate (percentage of hard samples added to train set per epoch).")
parser.add_option("-v", "--verbose", action="store_true", dest="verbose", default=False)
parser.add_option("-V", "--very-verbose", action="store_true", dest="very_verbose", default=False)

(options, args) = parser.parse_args()

#Mandatory arguments
if len(args) < 1:
    print("Error: no data folder(s) specified.")
    parser.print_usage()
    sys.exit(2)
data_folders = args
window_code = options.window_code
window_size = (int(window_code[0]), int(window_code[1]), int(window_code[2])) # converting window string code to int tuple
features_string = options.features_string
pca_components_total = options.pca_components_total
max_epochs = options.max_epochs
convergence_threshold = options.convergence_threshold
initial_train_size = options.initial_train_size
learning_rate = options.learning_rate
lss_util.verbose = options.verbose
lss_util.very_verbose = options.very_verbose

################################################################
# Reading supervised liver data
t0 = time()
printv("Loading data from {}... ".format(data_folders), end="", flush=True)

raw_dataset = list(chain.from_iterable([lss_data.load_folder(data_folder) for data_folder in data_folders]))

printv("Done in {:.3f}s. Loaded {} patients.".format(time()-t0, len(raw_dataset)))

################################################################
# Assembling dataset of patches
printv("Dividing dataset into patches of size {}... ".format(window_size), end="", flush=True)

patch_dataset, patch_stats = lss_data.patch_dataset(raw_dataset, window_size)

printv("Done in {:.3f}s.".format(time()-t0))

# Print stats on dataset
print("Total of patches: {}".format(patch_stats['total']))
print("Non-target patches: {} ({:.2f}%)".format(patch_stats['per_class'][0], (patch_stats['per_class'][0]*100/patch_stats['total'])))
print("    Target patches: {} ({:.2f}%)".format(patch_stats['per_class'][1], (patch_stats['per_class'][1]*100/patch_stats['total'])))

################################################################
# Extracting features
printv("Extracting features {}... ".format(features_string), end="", flush=True)

feature_dataset = lss_features.extract_features(patch_dataset, features_string)

printv("Done in {:.3f}s. {} features extracted.".format(time()-t0, len(feature_dataset[0]['features'])))

################################################################
# Assembling training and test sets

################################################################
# Training the SVM



################################################################
# Final report and model storing