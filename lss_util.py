"""
Script containing utility functions for the Liver SVM Segmentator.
"""
################################################################
# Utility functions

verbose = False
def printv(string, end='\n', flush=False):
    """
    Function that prints a message only if the global verbose flag is up.
    """
    if verbose or very_verbose:
        print(string, end=end, flush=flush)

very_verbose = False
def printvv(string, end='\n', flush=False):
    """
    Function that prints a message only if the global verbose flag is up.
    """
    if very_verbose:
        print(string, end=end, flush=flush)