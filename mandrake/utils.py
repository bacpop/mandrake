# vim: set fileencoding=<utf-8> :
# Copyright 2021 John Lees

'''Shared methods'''

import numpy as np

# Transforms the provided array to normalise and centre it
def norm_and_centre(array):
    means = np.mean(array, axis=0)
    array -= means
    scales = np.std(array, axis=0)
    array /= scales