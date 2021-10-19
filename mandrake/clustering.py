# vim: set fileencoding=<utf-8> :
# Copyright 2020 John Lees and Gerry Tonkin-Hill

'''Methods for clustering using an embedding as input
'''

import hdbscan
from collections import defaultdict
import numpy as np
import pandas as pd


def runHDBSCAN(embedding):
    embedding_scaled = _scale_and_centre(embedding)
    hdb = hdbscan.HDBSCAN(algorithm='boruvka_balltree',
                          min_cluster_size=2,
                          min_samples=2,
                          cluster_selection_epsilon=0.02,
                          allow_single_cluster=True
                          ).fit(embedding_scaled)
    return hdb.labels_


def write_hdbscan_clusters(clusters, labels, output_prefix):
    d = defaultdict(list)
    for label, cluster in zip(labels, clusters):
        d['id'].append(label)
        d['hdbscan_cluster__autocolour'].append(cluster)
    pd.DataFrame(data=d).to_csv(output_prefix + ".embedding_hdbscan_clusters.csv",
                                index=False)


# Internal functions


def _scale_and_centre(array):
    means = np.mean(array, axis=0)
    array_scaled = array - means
    scales = 0.5 * (np.max(array_scaled, axis=0) -
                    np.min(array_scaled, axis=0))
    array_scaled /= scales
    return(array_scaled)
