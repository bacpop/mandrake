# vim: set fileencoding=<utf-8> :
# Copyright 2020 John Lees and Gerry Tonkin-Hill

import os
import numpy as np

from pathoSCE.utils import readRfile, distVec, distVecCutoff

DATA_DIR = 'tests/data'
RFILE = os.path.join(DATA_DIR, 'rfiles.txt')

def test_readRfile():
    expected_names = ['12673_8_24', '12673_8_26', '12673_8_27', '12673_8_28', '12673_8_29']
    expected_sequences = [['./assemblies/12673_8#24.contigs_velvet.fa', './reads/12673_8#24.fq'],
                          ['./assemblies/12673_8#26.contigs_velvet.fa'],
                          ['./assemblies/12673_8#27.contigs_velvet.fa'],
                          ['./assemblies/12673_8#28.contigs_velvet.fa'],
                          ['./assemblies/12673_8#29.contigs_velvet.fa']
                         ]
	
    names, sequences = readRfile(RFILE)
    print(sequences)
    assert names == expected_names
    assert sequences == expected_sequences

def test_distVecCutoff():
    I, J = distVec(3)
    assert np.all(I == np.array([0, 0, 1]))
    assert np.all(J == np.array([1, 2, 2]))

def test_distVecCutoff():
    I, J, P = distVecCutoff(np.array([0.1, 0.1, 1]), 3, 0.5)
    assert np.all(I == np.array([0, 0]))
    assert np.all(J == np.array([1, 2]))
    assert np.all(P == np.array([0.1, 0.1]))


