# vim: set fileencoding=<utf-8> :
# Copyright 2020 Gerry Tonkin-Hill

'''pairsnp functions for determining pairwise SNP distances
from a multiple sequence file'''

import sys
import subprocess
import numpy as np
from scipy.sparse import coo_matrix


def checkPairsnpVersion():
    """Checks that pairsnp can be run, and returns version

    Returns:
        version (str)
            Version string
    """
    p = subprocess.Popen([pairsnp_exe + ' --version'], shell=True, stdout=subprocess.PIPE)
    version = 0
    for line in iter(p.stdout.readline, ''):
        if line != '':
            version = line.rstrip().decode().split(" ")[1]
            break

    return version

# from BioPython
def read_fasta(fp):
    name, seq = None, []
    for line in fp:
        line = line.rstrip()
        if line.startswith(">"):
            if name: yield (name, ''.join(seq))
            name, seq = line, []
        else:
            seq.append(line)
    if name: yield (name, ''.join(seq))

def runPairsnp(pairsnp_exe, msaFile, output, kNN=None, threshold=None, threads=1):
    """Runs pairsnp in sparse output mode with the option of supplying a distance or kNN cutoff

    Args:
        msaFile (str)
            Multiple sequence alignment
        output (str)
            Prefix for output files
        threshold (float)
            Proportion of alignment allowed to differ. Converted to a SNP distance threshold (optional)
        threads (int)
            Number of threads to use when running pairsnp (default=1)

    Returns:
        distMatrix (csr_matrix)
            Sparse pairwise snp distance matrix
        seqNames (list)
            A list of sequence names in the same order as the distance matrix
    """

    if (kNN is None) and (threshold is None):
        sys.stderr.write("Can not specify both kNN and threshold with pairsnp!\n")
        sys.exit(1)

    # get alignment length
    with open(msaFile, 'r') as msa:
        aln_len = len(next(read_fasta(msa))[1])

    # run pairsnp command
    cmd = pairsnp_exe + ' -s'
    if threshold is not None:
        distance = int(np.floor(threshold*aln_len))
        cmd += ' -d ' + str(distance)
    if kNN is not None:
        cmd += ' -k ' + str(kNN)
    cmd += ' -t ' + str(threads)
    cmd += ' ' + msaFile

    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    line_iter = iter(p.stdout)

    # process result
    seqNames = next(line_iter).decode("utf-8").strip().split("\t")[1:]

    distances = np.genfromtxt(line_iter, dtype=np.int32, delimiter="\t")

    if len(distances)<=2:
        sys.stderr.write("Distance threshold is too strict, less than 3 pairs passed!\n")
        sys.exit(1)

    distMatrix = coo_matrix(((distances[:,2]+0.1)/aln_len, (distances[:,0], distances[:,1])),
        shape=(len(seqNames), len(seqNames)))

    return distMatrix, seqNames
