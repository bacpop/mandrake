# vim: set fileencoding=<utf-8> :
# Copyright 2020 Gerry Tonkin-Hill

"""pairsnp functions for determining pairwise SNP distances
from a multiple sequence file"""

import sys, os
import numpy as np

# C++ extensions
sys.path.insert(0, os.path.dirname(__file__) + "/../build/lib.linux-x86_64-3.8")
from SCE import pairsnp

# from BioPython
def read_fasta(fp):
    name, seq = None, []
    for line in fp:
        line = line.rstrip()
        if line.startswith(">"):
            if name:
                yield (name, "".join(seq))
            name, seq = line, []
        else:
            seq.append(line)
    if name:
        yield (name, "".join(seq))


def runPairsnp(msaFile, kNN=None, threshold=None, threads=1):
    """Runs pairsnp with the option of supplying a distance or kNN cutoff

    Args:
        msaFile (str)
            Multiple sequence alignment
        kNN (int)
            Number of nearest neighbours to return for each sample
        threshold (float)
            Proportion of alignment allowed to differ. Converted to a SNP distance threshold (optional)
        threads (int)
            Number of threads to use when running pairsnp (default=1)

    Returns:
        I (np.array)
            integer array of row indices
        J (np.array)
            integer array of column indices
        dist (np.array)
            array of SNP distances
        names (list)
            list of sample names taken from the fasta headers
    """

    # run some checks on the parameters
    if not os.path.isfile(msaFile):
        raise ValueError("MSA file does not exist!")
    if kNN is not None:
        if not (isinstance(kNN, int) and (kNN > 0)):
            raise ValueError("invalid value for kNN!")
    if threshold is not None:
        if not (isinstance(threshold, float) and (threshold > 0) and (threshold <= 1)):
            raise ValueError("invalid value for threshold!")
    if not (isinstance(threads, int) and (threads > 0)):
        raise ValueError("invalid value for threads!")

    # determine the max snp distance and alignment length
    with open(msaFile, "r") as infile:
        name, seq = next(read_fasta(infile))
        seq_len = len(seq)

    if threshold is not None:
        dist = int(threshold * seq_len) + 1
    else:
        dist = -1

    # run pairsnp
    I, J, dist, names = pairsnp(fasta=msaFile, n_threads=threads, dist=dist, knn=kNN)

    return (np.array(I), np.array(J), np.array(dist) / seq_len, names)
