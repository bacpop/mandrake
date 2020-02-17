# vim: set fileencoding=<utf-8> :
# Copyright 2020 John Lees and Gerry Tonkin-Hill

'''Methods for calculating distances from sequence input'''

import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix
from scipy.spatial.distance import squareform, pdist

# C++ extensions
import pp_sketchlib

from .pairsnp import runPairsnp
from .sketchlib import readDBParams, getSeqsInDb
from .utils import readRfile

def accessoryDists(accessory_file, sparse):
    acc_mat = pd.read_csv(accessory_file, sep="\t", header=0, index_col=0, dtype=np.bool_)
    names = list(acc_mat.columns())
    if sparse:
        P = _sparseJaccard(acc_mat.values)
    else:
        P = _denseJaccard(acc_mat.values)

    return P, names

def pairSnpDists(pairsnp_exe, alignment, output, threshold, cpus):
    # alignment
    P, names = runPairsnp(pairsnp_exe,
                          alignment, 
                          output, 
                          threshold=threshold, 
                          threads=cpus)
    # TODO: keep as sparse if possible later on.
    P = squareform(P.todense(), force='tovector', checks=False)
    return P, names

def sketchlibDists(sequence_file, output, kmers, sketch_size, min_count, dist_col, cpus):
    names, sequences = readRfile(sequence_file)
    P = pp_sketchlib.constructAndQuery(output, 
                                        names, 
                                        sequences, 
                                        kmers, 
                                        int(round(sketch_size/64)), 
                                        min_count, 
                                        cpus)[:,dist_col]
    return P, names

def sketchlibDbDists(sketch_db, default_kmers, default_sketchsize, dist_col, cpus):
    names = getSeqsInDb(sketch_db + ".h5")
    kmers, sketch_size = readDBParams(sketch_db + ".h5", default_kmers, int(round(default_sketchsize/64)))
    P = pp_sketchlib.queryDatabase(sketch_db, sketch_db, names, names, kmers, cpus)[:,dist_col]
    return P, names

# Internal functions

def _sparseJaccard(m):
    sm = csc_matrix(m)
    cTT = sm*sm.transpose()
    cTT = cTT.todense()
    temp = 1-np.eye(sm.shape[0])
    di = np.diag(cTT)
    d = 1-(cTT/((temp*di).transpose() + temp*di - cTT))
    return squareform(d, force='tovector', checks=False)

def _denseJaccard(m):
    return(pdist(m, 'jaccard'))