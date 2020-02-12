# vim: set fileencoding=<utf-8> :
# Copyright 2020 John Lees and Gerry Tonkin-Hill

import numpy as np
from scipy.sparse import csc_matrix
from scipy.spatial.distance import squareform, pdist

def sparseJaccard(m):
    sm = csc_matrix(m)
    cTT = sm*sm.transpose()
    cTT = cTT.todense()
    temp = 1-np.eye(sm.shape[0])
    di = np.diag(cTT)
    d = 1-(cTT/((temp*di).transpose() + temp*di - cTT))
    return squareform(d, force='tovector', checks=False)

def denseJaccard(m):
    return(pdist(m, 'jaccard'))
