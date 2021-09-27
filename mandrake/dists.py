# vim: set fileencoding=<utf-8> :
# Copyright 2020 John Lees and Gerry Tonkin-Hill

'''Methods for calculating distances from sequence input'''

import numpy as np
import pandas as pd
from sklearn.neighbors import kneighbors_graph

# C++ extensions
import sys, os
sys.path.insert(0, os.path.dirname(__file__) + '/../build/lib.linux-x86_64-3.9')
sys.path.insert(0, os.path.dirname(__file__) + '/../build/lib.macosx-10.9-x86_64-3.9')
import pp_sketchlib

from .pairsnp import runPairsnp
from .sketchlib import get_kmer_sizes, get_seqs_in_db

def accessoryDists(accessory_file, kNN, threshold, cpus):
    acc_mat = pd.read_csv(accessory_file, sep="\t", header=0, index_col=0)
    names = list(acc_mat.columns)

    if kNN is None:
        kNN = len(names)

    sp = kneighbors_graph(X=acc_mat, n_neighbors=kNN, 
        metric = 'jaccard', mode = 'distance',
        include_self=False, n_jobs=cpus).tocoo()

    if threshold is not None:
        index = []
        for i, d in enumerate(sp.data):
            if d<threshold:
                index.append(i)
        index = np.array(index)
        sp.row = sp.row[index]
        sp.col = sp.col[index]
        sp.data = sp.data[index]

    return sp.row, sp.col, sp.data, names

def pairSnpDists(alignment, threshold, kNN, cpus):
    I, J, dists, names = runPairsnp(alignment,
                          kNN=kNN,
                          threshold=threshold,
                          threads=cpus)
    return I, J, dists, names

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