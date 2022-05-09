# vim: set fileencoding=<utf-8> :
# Copyright 2020 John Lees and Gerry Tonkin-Hill

'''Methods for calculating distances from sequence input'''

import re
import numpy as np
import pandas as pd
from sklearn.neighbors import kneighbors_graph

# C++ extensions
import pp_sketchlib

from .pairsnp import runPairsnp
from .sketchlib import get_kmer_sizes, get_seqs_in_db


def accessoryDists(accessory_file, kNN, threshold, cpus):
    acc_mat = pd.read_csv(accessory_file, sep="\t", header=0, index_col=0)
    names = list(acc_mat.columns)

    if kNN <= 0:
        kNN = len(names) - 1

    sp = kneighbors_graph(X=acc_mat.T, n_neighbors=kNN,
                          metric='jaccard', mode='distance',
                          include_self=False, n_jobs=cpus).tocoo()

    if threshold > 0:
        index = []
        for i, d in enumerate(sp.data):
            if d < threshold:
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

    sketchlib_version = re.search(r"(\d+)\.(\d+)\.(\d+)", pp_sketchlib.version)
    if sketchlib_version and int(sketchlib_version.group(1)) >= 2:
        # v2 of sketchlib supports 'true' sparse query which reduces distance
        # matrix on the fly
        if (len(kmers) == 1):
            jaccard = True
        else:
            jaccard = False
        if threshold > 0:
            raise ValueError("Use kNN with --sketches")
        I, J, dists = pp_sketchlib.querySelfSparse(ref_db_name=sketch_db,
                                                   rList=names,
                                                   klist=kmers,
                                                   random_correct=True,
                                                   jaccard=jaccard,
                                                   kNN=kNN,
                                                   dist_cutoff=0,
                                                   dist_col=dist_col,
                                                   num_threads=cpus,
                                                   use_gpu=use_gpu,
                                                   device_id=device_id)
    else:
        # older versions of sketchlib do a dense query then sparsify the
        # return. Ok for smaller data, but runs out of memory on big datasets
        # sketchlib API needs positive int for kNN
        if kNN < 0:
            kNN = 0
        if threshold <= 0:
            threshold = 0.0
        I, J, dists = pp_sketchlib.queryDatabaseSparse(ref_db_name=sketch_db,
                                                       query_db_name=sketch_db,
                                                       rList=names,
                                                       qList=names,
                                                       klist=kmers,
                                                       random_correct=True,
                                                       dist_cutoff=threshold,
                                                       kNN=kNN,
                                                       core=(dist_col == 0),
                                                       num_threads=cpus,
                                                       use_gpu=use_gpu,
                                                       device_id=device_id)

    return I, J, dists, names
