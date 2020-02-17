# vim: set fileencoding=<utf-8> :
# Copyright 2020 John Lees and Gerry Tonkin-Hill

'''Methods for clustering on embedding
'''

import hdbscan

def runHDBSCAN(embedding):
    hdb = hdbscan.HDBSCAN(algorithm='boruvka_balltree',
                     min_cluster_size = 4,
                     min_samples = 2,
                     cluster_selection_epsilon = 0.2
                     ).fit(embedding)
    return hdb.labels_

