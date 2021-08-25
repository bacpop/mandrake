# vim: set fileencoding=<utf-8> :
# Copyright 2020 John Lees and Gerry Tonkin-Hill

'''Methods for calculating distances from sequence input'''

import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix, coo_matrix
from scipy.spatial.distance import squareform, pdist
from sklearn.neighbors import NearestNeighbors


# C++ extensions
import pp_sketchlib

from .pairsnp import runPairsnp
from .sketchlib import readDBParams, getSeqsInDb
from .utils import readRfile

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

def sketchlibDists(sequence_file, output, kmers, sketch_size, min_count, dist_col, kNN, threshold, cpus):
    names, sequences = readRfile(sequence_file)
    P = pp_sketchlib.constructAndQuery(output,
                                        names,
                                        sequences,
                                        kmers,
                                        int(round(sketch_size/64)),
                                        min_count,
                                        cpus)[:,dist_col]
    # TODO: replace with sparse version
    if kNN is not None:
        P = _KNN_conv(P, k=kNN)
    if threshold is not None:
        P= _threshold_conv(P, threshold=threshold)
    else:
        P = coo_matrix(squareform(P, force='tomatrix', checks=False))

    return P, names

def sketchlibDbDists(sketch_db, default_kmers, default_sketchsize, dist_col, kNN, threshold, cpus):
    names = getSeqsInDb(sketch_db + ".h5")
    kmers, sketch_size = readDBParams(sketch_db + ".h5", default_kmers, int(round(default_sketchsize/64)))
    P = pp_sketchlib.queryDatabase(sketch_db, sketch_db, names, names, kmers, cpus)[:,dist_col]

    # TODO: replace with sparse version
    if kNN is not None:
        P = _KNN_conv(P, k=kNN)
    if threshold is not None:
        P= _threshold_conv(P, threshold=threshold)
    else:
        P = coo_matrix(squareform(P, force='tomatrix', checks=False))

    return P, names

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

# TODO: Remove these functions. This is a temporary function used until we've updated the other distance functions
def _KNN_conv(P, k=None):
    P = squareform(P, force='tomatrix', checks=False)
    for i in range(P.shape[0]):
        m = np.max(P[i,np.argpartition(P[i,:], k)])
        P[i,P[i,:]>m] = 0

    P = coo_matrix(P)
    return(P)

def _threshold_conv(P, threshold=None):
    P = squareform(P, force='tomatrix', checks=False)
    P[P<=threshold] = 0
    P = coo_matrix(P)
    return(P)