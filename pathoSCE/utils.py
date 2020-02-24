# vim: set fileencoding=<utf-8> :
# Copyright 2020 John Lees

'''Utilities for file reading and vector generation'''

import sys, os
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
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
    D = csr_matrix(D)
    nsamples = D.shape[0]
    # print(D.size)
    # calculate probabilities row by row
    conditional_P = np.empty(D.size, dtype=np.float32)
    j=0
    for i in range(nsamples):
        temp = np.zeros((1, len(D[i,:].data)+1),  dtype=np.float32)
        temp[0,1:] = D[i,:].data
        temp = ut._binary_search_perplexity(temp, perplexity, False)[0][1:]
        conditional_P[j:(j+temp.size)] = temp
        j+=temp.size

    P = csr_matrix((conditional_P, D.indices,
                        D.indptr),
                    shape=(nsamples, nsamples)).tocoo()

    P = P + P.T 
    # Normalize the joint probability distribution
    sum_P = np.maximum(P.sum(), MACHINE_EPSILON)
    P /= sum_P

    return(P.tocoo())