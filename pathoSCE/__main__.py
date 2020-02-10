# Copyright 2019 John Lees

'''Wrapper around sketch functions'''

import os, sys

import numpy as np

import pp_sketchlib
from pairsnp import calculate_snp_matrix, calculate_distance_matrix
from SCE import wtsne

from .__init__ import __version__

def iterDistRows(refSeqs, querySeqs, self=True):
    """Gets the ref and query ID for each row of the distance matrix

    Returns an iterable with ref and query ID pairs by row.

    Args:
        refSeqs (list)
            List of reference sequence names.
        querySeqs (list)
            List of query sequence names.
        self (bool)
            Whether a self-comparison, used when constructing a database.

            Requires refSeqs == querySeqs

            Default is True
    Returns:
        ref, query (str, str)
            Iterable of tuples with ref and query names for each distMat row.
    """
    if self:
        if refSeqs != querySeqs:
            raise RuntimeError('refSeqs must equal querySeqs for db building (self = true)')
        for i, ref in enumerate(refSeqs):
            for j in range(i + 1, len(refSeqs)):
                yield(refSeqs[j], ref)
    else:
        for query in querySeqs:
            for ref in refSeqs:
                yield(ref, query)

def get_options():
    import argparse

    description = 'Visualisation of genomic distances in pathogen populations'
    parser = argparse.ArgumentParser(description=description,
                                     prog='pathoSCE')

    modeGroup = parser.add_argument_group('Input file')
    mode = modeGroup.add_mutually_exclusive_group(required=True)
    mode.add_argument('--alignment',
                        default=None,
                        help='Work from an alignment')
    mode.add_argument('--accessory',
                        default=None,
                        help='Work from accessory genome presence/absence')
    mode.add_argument('--sequence',
                        default=None,
                        help='Work from assembly or read data')
    mode.add_argument('--sketches',
                        default=None,
                        help='Work from sketch data')

    io = parser.add_argument_group('Input/output')
    io.add_argument('--rfile',
                    help='Samples to sketch')
    io.add_argument('--ref-db',
                    help='Prefix of reference database file')
    io.add_argument('--query-db',
                    help='Prefix of query database file')

    kmerGroup = parser.add_argument_group('Kmer comparison options (for sequence input)')
    kmerGroup.add_argument('--min-k', default = 13, type=int, help='Minimum kmer length [default = 13]')
    kmerGroup.add_argument('--max-k', default = 29, type=int, help='Maximum kmer length [default = 29]')
    kmerGroup.add_argument('--k-step', default = 4, type=int, help='K-mer step size [default = 4]')
    kmerGroup.add_argument('--sketch-size', default=10000, type=int, help='Kmer sketch size [default = 10000]')
    kmerGroup.add_argument('--min-count', default=20, type=int, help='Minimum k-mer count from reads [default = 20]')

    other = parser.add_argument_group('Other')
    other.add_argument('--cpus',
                        type=int,
                        default=1,
                        help='Number of CPUs to use '
                             '[default = 1]')
    other.add_argument('--version', action='version',
                       version='%(prog)s '+__version__)

    return parser.parse_args()


def main():
    args = get_options()

    #***********************#
    #* Run seq -> distance *#
    #***********************#
    
    # alignment 
    sparse_matrix, consensus, seq_names = calculate_snp_matrix(fasta.file.name)
    distMat = calculate_distance_matrix(sparse_matrix, consensus, "dist", False)

    # accessory
    # TODO: read csv and calculate dists

    # sequence
    distMat = pp_sketchlib.constructAndQuery(ref_db, names, sequences, kmers, int(round(sketch_size/64)), min_count, cpus)

    # sketches
    pp_sketchlib.constructDatabase(ref_db, names, sequences, kmers, int(round(sketch_size/64)), min_count, cpus)
    distMat = pp_sketchlib.queryDatabase(ref_db, ref_db, rList, qList, kmers, cpus)

    #***********************#
    #* Save distances      *#
    #***********************#
    names = iterDistRows(rList, rList, True)
    sys.stdout.write("\t".join(['Query', 'Reference', 'Core', 'Accessory']) + "\n")
    for i, (ref, query) in enumerate(names):
        sys.stdout.write("\t".join([query, ref, str(distMat[i,0]), str(distMat[i,1])]) + "\n")

    #***********************#
    #* set up data for SCE *#
    #***********************#
    # TODO

    #***********************#
    #* run SCE             *#
    #***********************#
    embedding = wtsne(I, J, P, weights, maxIter, threads, nRepuSamp, eta0)

    #***********************#
    #* plot embedding      *#
    #***********************#
    # (both core and accessory if available)
    # static
    # TODO

    # dynamic
    # TODO

    sys.exit(0)

if __name__ == "__main__":
    main()