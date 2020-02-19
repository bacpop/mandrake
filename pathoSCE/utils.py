# vim: set fileencoding=<utf-8> :
# Copyright 2020 John Lees

'''Utilities for file reading and vector generation'''

import sys, os
import numpy as np
from numba import jit

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

# @jit(nopython=True)
# def distVecCutoff(P, length, cutoff):
#     dist_length = int(0.5*length*(length-1))
#     I_vec = np.empty((dist_length), dtype=np.int64)
#     J_vec = np.empty((dist_length), dtype=np.int64)
#     P_vec = np.empty((dist_length), dtype=np.float64)
    
#     counter = 0
#     included = 0
#     for i in range(length):
#         for j in range(i + 1, length):
#             if cutoff == None or P[counter] < cutoff:
#                 I_vec[counter] = i
#                 J_vec[counter] = j
#                 P_vec[counter] = P[counter]
#                 included +=1
#             counter += 1

#     return(I_vec[0:included], J_vec[0:included], P_vec[0:included])

# @jit(nopython=True)
# def distVec(length):
#     dist_length = int(0.5*length*(length-1))
#     I_vec = np.empty((dist_length), dtype=np.int64)
#     J_vec = np.empty((dist_length), dtype=np.int64)
    
#     counter = 0
#     for i in range(length):
#         for j in range(i + 1, length):
#             I_vec[counter] = i
#             J_vec[counter] = j
#             counter += 1

#     return(I_vec, J_vec)