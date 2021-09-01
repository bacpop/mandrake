# vim: set fileencoding=<utf-8> :
# Copyright 2020 John Lees and Gerry Tonkin-Hill

'''Methods for setting up and running SCE, saving
and loading results'''

import sys, os
from functools import partial
import numpy as np
import pandas as pd

# C++ extensions
sys.path.insert(0, os.path.dirname(__file__) + '/../build/lib.linux-x86_64-3.9')
sys.path.insert(0, os.path.dirname(__file__) + '/../build/lib.macosx-10.9-x86_64-3.9')
from SCE import wtsne
try:
    # Using doubles for now
    from SCE import wtsne_gpu_fp64 as wtsne_gpu
    gpu_fn_available = True
except ImportError:
    gpu_fn_available = False

# from .utils import distVec, distVecCutoff

# Run exits if fewer samples than this
MIN_SAMPLES = 100
MACHINE_EPSILON = np.finfo(np.double).eps

def save_input(I, J, dists, names, output_prefix):
    if (len(names) < MIN_SAMPLES):
        sys.stderr.write("Less than minimum number of samples used (" + str(MIN_SAMPLES) + ")\n")
        sys.stderr.write("Distances calculated, but not running SCE\n")
        sys.exit(1)

    pd.Series(names).to_csv(output_prefix + '.names.txt', sep='\n', header=False, index=False)

    _saveDists(output_prefix, I, J, dists)

def loadIJdist(npzfilename):
    npzfile = np.load(npzfilename)
    I = npzfile['I']
    J = npzfile['J']
    dists = npzfile['dists']
    return I, J, dists

def runSCE(I, J, dists, weight_file, names, SCE_opts):
    weights = np.ones((len(names)))
    if (weight_file):
        weights_in = pd.read_csv(weights, sep="\t", header=None, index_col=0)
        if (weights_in.index.symmetric_difference(names)):
            sys.stderr.write("Names in weights do not match sequences - using equal weights\n")
        else:
            intersecting_samples = weights_in.index.intersection(names)
            weights = weights_in.loc[intersecting_samples]

    # Set up function call with either CPU or GPU
    if SCE_opts['use_gpu'] and gpu_fn_available:
        sys.stderr.write("Running on GPU\n")
        wtsne_call = partial(wtsne_gpu,
                             perplexity = SCE_opts['perplexity'],
                             maxIter = SCE_opts['maxIter'],
                             blockSize = SCE_opts['blockSize'],
                             blockCount = SCE_opts['blockCount'],
                             nRepuSamp = SCE_opts['nRepuSamp'],
                             eta0 = SCE_opts['eta0'],
                             bInit = SCE_opts['bInit'],
                             n_threads = SCE_opts['cpus'],
                             device_id = SCE_opts['device_id'],
                             seed = 1)
    else:
        sys.stderr.write("Running on CPU\n")
        wtsne_call = partial(wtsne,
                             perplexity = SCE_opts['perplexity'],
                             maxIter = SCE_opts['maxIter'],
                             nRepuSamp = SCE_opts['nRepuSamp'],
                             eta0 = SCE_opts['eta0'],
                             bInit = SCE_opts['bInit'],
                             n_threads = SCE_opts['cpus'],
                             seed = 1)

    # Run embedding with C++ extension
    embedding = np.array(wtsne_call(I, J, dists, weights))
    embedding = embedding.reshape(-1, 2)
    return(embedding)

def saveEmbedding(embedding, output_prefix):
    np.savetxt(output_prefix + ".embedding.txt", embedding)

# Internal functions

def _saveDists(output_prefix, I, J, dists):
    np.savez(output_prefix, I=I, J=J, dists=dists)
