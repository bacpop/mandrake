# vim: set fileencoding=<utf-8> :
# Copyright 2020 John Lees and Gerry Tonkin-Hill

'''Methods for calculating distances from sequence input'''

import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix, coo_matrix
from sklearn.neighbors import NearestNeighbors

# C++ extensions
import sys, os
sys.path.insert(0, '/home/jlees/installs/pp-sketchlib/build/lib.linux-x86_64-3.8')
import pp_sketchlib

from .pairsnp import runPairsnp
from .sketchlib import get_kmer_sizes, get_seqs_in_db

def accessoryDists(accessory_file, kNN, threshold):
    acc_mat = pd.read_csv(accessory_file, sep="\t", header=0, index_col=0)
    names = list(acc_mat.columns)
    if kNN is not None:
        I, J, dists = _kNNJaccard(acc_mat, kNN)
    else:
        I, J, dists = _sparseJaccard(acc_mat.values, threshold)

    return I, J, dists, names

# TODO: need I, J, dists, names return
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
    names = get_seqs_in_db(sketch_db + ".h5")
    kmers = get_kmer_sizes(sketch_db + ".h5")
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

    sp_coo = coo_matrix(d)
    return(sp_coo.data, sp_coo.row, sp_coo.col)

def _kNNJaccard(m, k):
    neigh = NearestNeighbors(n_neighbors=k, metric='jaccard')
    neigh.fit(m)
    d = neigh.kneighbors(m)

    return(d[0].flatten(), np.repeat(np.arange(m.shape[0]), 2), d[1].flatten())
