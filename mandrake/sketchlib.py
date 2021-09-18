# vim: set fileencoding=<utf-8> :
# Copyright 2020 John Lees

'''Sketchlib functions for database construction'''

import os
import sys
import subprocess
import numpy as np

import h5py

def get_kmer_sizes(dbPrefix):
    """Get kmers lengths from existing database

    Args:
        dbPrefix (str)
            Prefix for sketch DB files
    Returns:
        kmers (list)
            List of k-mer lengths used in database
    """
    ref_db = h5py.File(dbPrefix, 'r')
    db_kmer_sizes = []
    for sample_name in list(ref_db['sketches'].keys()):
        kmer_size = ref_db['sketches/' + sample_name].attrs['kmers']
        if len(db_kmer_sizes) == 0:
            db_kmer_sizes = kmer_size
        elif np.any(kmer_size != db_kmer_sizes):
            sys.stderr.write("Problem with database; kmer lengths inconsistent: " +
                             str(kmer_size) + " vs " + str(db_kmer_sizes) + "\n")
            sys.exit(1)

    db_kmer_sizes.sort()
    return list(db_kmer_sizes)

def get_seqs_in_db(dbname):
    """Return an array with the sequences in the passed database

    Args:
        dbname (str)
            Sketches database filename

    Returns:
        seqs (list)
            List of sequence names in sketch DB
    """
    seqs = []
    ref = h5py.File(dbname, 'r')
    for sample_name in list(ref['sketches'].keys()):
        seqs.append(sample_name)

    return seqs


