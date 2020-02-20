# vim: set fileencoding=<utf-8> :
# Copyright 2020 John Lees and Gerry Tonkin-Hill

'''Methods for setting up and running SCE, saving
and loading results'''

import sys
from functools import partial
import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform
from sklearn.manifold.t_sne import _joint_probabilities

# C++ extensions
from SCE import wtsne
try:
    from SCE import wtsne_gpu
    gpu_available = True
except ImportError:
    gpu_available = False

from .utils import distVec, distVecCutoff

# Run exits if fewer samples than this
MIN_SAMPLES = 100
DEFAULT_THRESHOLD = 1.0

def generateIJP(names, output_prefix, threshold, P, preprocessing, perplexity):
    if (len(names) < MIN_SAMPLES):
        sys.stderr.write("Less than minimum number of samples used (" + str(MIN_SAMPLES) + ")\n")
        sys.stderr.write("Distances calculated, but not running SCE\n")
        sys.exit(1)
        
    pd.Series(names).to_csv(output_prefix + '.names.txt', sep='\n', header=False, index=False)
    if threshold == DEFAULT_THRESHOLD:
        I, J = distVec(len(names))
    else:
        I, J, P = distVecCutoff(P, len(names), threshold)
        
    # convert to similarity
    P = distancePreprocess(P, preprocessing, perplexity)

    # SCE needs symmetric distances too
    I_stack = np.concatenate((I, J), axis=None)
    J_stack = np.concatenate((J, I), axis=None)
    I = I_stack
    J = J_stack
    P = np.concatenate((P, P), axis=None)

    _saveDists(output_prefix, I, J, P)
    return(I, J, P)

def distancePreprocess(P, preprocessing, perplexity):
    if preprocessing:
        # entropy preprocessing
        P = _joint_probabilities(squareform(P, force='tomatrix', checks=False), 
                                    desired_perplexity=perplexity, 
                                    verbose=0)
    else:
        P = 1 - P/np.max(P)
    return P

def loadIJP(npzfilename):
    npzfile = np.load(npzfilename)
    I = npzfile['I']
    J = npzfile['J']
    P = npzfile['P']
    return I, J, P

def runSCE(I, J, P, weight_file, names, SCE_opts, use_gpu=False):
    weights = np.ones((len(names)))
    if (weight_file):
        weights_in = pd.read_csv(weights, sep="\t", header=None, index_col=0)
        if (weights_in.index.symmetric_difference(names)):
            sys.stderr.write("Names in weights do not match sequences - using equal weights\n")
        else:
            intersecting_samples = weights_in.index.intersection(names)
            weights = weights_in.loc[intersecting_samples]
    
    # Set up function call with either CPU or GPU
    if use_gpu and gpu_available:
        wtsne_call = partial(wtsne_gpu, 
                             maxIter = SCE_opts['maxIter'], 
                             blockSize = 128, blockCount = 128,
                             nRepuSamp = SCE_opts['nRepuSamp'],
                             eta0 = SCE_opts['eta0'],
                             bInit = SCE_opts['bInit'])
    else:
        wtsne_call = partial(wtsne,
                             maxIter = SCE_opts['maxIter'], 
                             workerCount = SCE_opts['cpus'],
                             nRepuSamp = SCE_opts['nRepuSamp'],
                             eta0 = SCE_opts['eta0'],
                             bInit = SCE_opts['bInit'])

    # Run embedding with C++ extension
    embedding = np.array(wtsne_call(I, J, P, weights))
    embedding = embedding.reshape(-1, 2)
    return(embedding)

def saveEmbedding(embedding, output_prefix):
    np.savetxt(output_prefix + ".embedding.txt", embedding)

# Internal functions

def _saveDists(output_prefix, I, J, P):
    np.savez(output_prefix, I=I, J=J, P=P)
