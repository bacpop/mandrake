# vim: set fileencoding=<utf-8> :
# Copyright 2020 John Lees and Gerry Tonkin-Hill

'''Methods for calculating distances from sequence input'''

import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix, coo_matrix
from scipy.spatial.distance import squareform, pdist
from sklearn.neighbors import NearestNeighbors

# C++ extensions
import sys, os
sys.path.insert(0, '/home/jlees/installs/pp-sketchlib/build/lib.linux-x86_64-3.8')
import pp_sketchlib

from .pairsnp import runPairsnp
from .sketchlib import readDBParams, getSeqsInDb

def accessoryDists(accessory_file, sparse, kNN, threshold):
    acc_mat = pd.read_csv(accessory_file, sep="\t", header=0, index_col=0)
    names = list(acc_mat.columns)
    if kNN is not None:
        P = _kNNJaccard(acc_mat, kNN)
    elif sparse:
        P = _sparseJaccard(acc_mat.values)
    else:
        P = _denseJaccard(acc_mat.values)
    return P, names

def pairSnpDists(pairsnp_exe, alignment, output, threshold, kNN, cpus):
    # alignment
    P, names = runPairsnp(pairsnp_exe,
                          alignment,
                          output,
                          threshold=threshold,
                          kNN=kNN,
                          threads=cpus)

    return P, names

def sketchlibDists(sketch_db, dist_col, kNN, threshold, cpus, use_gpu, device_id):
    names = getSeqsInDb(sketch_db + ".h5")
    kmers, sketch_size = readDBParams(sketch_db + ".h5")
    I, J, dists = pp_sketchlib.queryDatabaseSparse(sketch_db,
                                                sketch_db,
                                                names,
                                                names,
                                                kmers,
                                                True,
                                                threshold,
                                                kNN,
                                                dist_col == 0,
                                                cpus,
                                                use_gpu,
                                                device_id)

    return I, J, dists, names

# Internal functions

def _sparseJaccard(m, threshold=None):
    sm = csc_matrix(m)
    cTT = sm*sm.transpose()
    cTT = cTT.todense()
    temp = 1-np.eye(sm.shape[0])
    di = np.diag(cTT)
    d = 1-(cTT/((temp*di).transpose() + temp*di - cTT))
    if threshold is not None:
        d[d>threshold] = 0
    return coo_matrix(d)

def _denseJaccard(m, threshold=None):
    d = squareform(pdist(m, 'jaccard'))
    if threshold is not None:
        d[d>threshold] = 0
    return(coo_matrix(d))

def _kNNJaccard(m, k):
    neigh = NearestNeighbors(n_neighbors=k, metric='jaccard')
    neigh.fit(m)
    d = neigh.kneighbors(m)

    d = coo_matrix((d[0].flatten(),
            (np.repeat(np.arange(m.shape[0]),2), d[1].flatten())),
            shape=(m.shape[0], m.shape[0]))

    return(d)
