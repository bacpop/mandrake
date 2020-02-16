# vim: set fileencoding=<utf-8> :
# Copyright 2020 Gerry Tonkin-Hill

'''pairsnp functions for determining pairwise SNP distances from a multiple sequence file'''

import subprocess
import numpy as np
from scipy.sparse import csr_matrix

pairsnp_exe = "pairsnp"

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


def runPairsnp(msaFile, output, distance=np.Inf, threads=1):
    """Runs pairsnp in sparse output mode with the option of supplying a distance cutoff

    Args:
        msaFile (str)
            Multiple sequence alignment
        output (str)
            Prefix for output files
        distance (int)
            Pairwise SNP distance threshold (optional)
        threads (int)
            Number of threads to use when running pairsnp (default=1)

    Returns:
        distMatrix (csr_matrix)
            Sparse pairwise snp distance matrix
        seqNames (list)
            A list of sequence names in the same order as the distance matrix
    """
    
    # run pairsnp command
    cmd = pairsnp_exe + ' -s'
    if distance<np.Inf:
        cmd += ' -d ' + str(distance)
    cmd += ' -t ' + str(threads)
    cmd += ' ' + msaFile

    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    line_iter = iter(p.stdout)

    # process result
    seqNames = next(line_iter).decode("utf-8").strip().split("\t")[1:]

    distances = np.genfromtxt(line_iter, dtype=int, delimiter="\t")

    distMatrix = csr_matrix((distances[:,2], (distances[:,0], distances[:,1])), 
        shape=(len(seqNames), len(seqNames)))

    return distMatrix, seqNames
