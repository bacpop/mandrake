# vim: set fileencoding=<utf-8> :
# Copyright 2020 John Lees and Gerry Tonkin-Hill

'''Methods for clustering using an embedding as input
'''

import hdbscan
from collections import defaultdict
import pandas as pd


def runHDBSCAN(embedding):
    hdb = hdbscan.HDBSCAN(algorithm='boruvka_balltree',
                     min_cluster_size = 2,
                     min_samples = 2,
                     cluster_selection_epsilon = 0.1,
                     allow_single_cluster=True
                     ).fit(embedding)
    return hdb.labels_


def write_hdbscan_clusters(clusters, labels, output_prefix):
    d = defaultdict(list)
    for label, cluster in zip(labels, clusters):
        d['id'].append(label)
        d['hdbscan_cluster__autocolour'].append(cluster)
    pd.DataFrame(data=d).to_csv(output_prefix + ".hdbscan_clusters.csv",
                                index = False)