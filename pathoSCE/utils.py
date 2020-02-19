# vim: set fileencoding=<utf-8> :
# Copyright 2020 John Lees

'''Utilities for file reading and vector generation'''

import sys, os
import numpy as np
from numba import jit
from sklearn.manifold import _utils as ut
MACHINE_EPSILON = np.finfo(np.double).eps

def readRfile(rFile):
    """Reads in files for sketching. Names and sequence, tab separated

    Args:
        rFile (str)
            File with locations of assembly files to be sketched

    Returns:
        names (list)
            Array of sequence names
        sequences (list of lists)
            Array of sequence files
    """
    names = []
    sequences = []
    with open(rFile, 'r') as refFile:
        for refLine in refFile:
            rFields = refLine.rstrip().split("\t")
            if len(rFields) < 2:
                sys.stderr.write("Input reference list is misformatted\n"
                                 "Must contain sample name and file, tab separated\n")
                sys.exit(1)
            
            names.append(rFields[0])
            sequences.append(list(rFields[1:]))
            #sample_files = []
            #for sequence in rFields[1:]:
            #    sample_files.append(sequence)
            #sequences.append(sample_files)

    if len(set(names)) != len(names):
        sys.stderr.write("Input contains duplicate names! All names must be unique\n")
        sys.exit(1)

    return (names, sequences)

def sparse_joint_probabilities(D, perplexity):
    nsamples = D.shape[0]
    # calculate probabilities row by row
    for i in range(nsamples):
        temp = D[i,:].data
        temp = np.full((1, len(temp)), temp, dtype=np.float32)
        D[i,:].data = ut._binary_search_perplexity(
                        temp, perplexity, False)
    D = D + D.T
    # Normalize the joint probability distribution
    sum_P = np.maximum(D.sum(), MACHINE_EPSILON)
    D /= sum_P
    return(D)