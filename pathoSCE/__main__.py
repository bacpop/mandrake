# Copyright 2019 John Lees

'''Main control function for pathoSCE'''

import os, sys

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from numba import jit

import pp_sketchlib
from SCE import wtsne

from .__init__ import __version__

from .sketchlib import readDBParams
from .sketchlib import getSeqsInDb

from .plot import plotSCE

from .utils import distVec
from .utils import readRfile

MIN_SAMPLES = 100

def get_options():
    import argparse

    description = 'Visualisation of genomic distances in pathogen populations'
    parser = argparse.ArgumentParser(description=description,
                                     prog='pathoSCE')

    modeGroup = parser.add_argument_group('Input type')
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
    mode.add_argument('--distances',
                        default=None,
                        help='Work from pre-calculated distances')

    distanceGroup = parser.add_argument_group('Output options')
    distanceGroup.add_argument('--output', default="pathoSCE", type=str, help='Prefix for output files [default = "pathoSCE"]')
    distanceGroup.add_argument('--threshold', default=1, type=float, help='Maximum distance to consider [default = 1')

    sceGroup = parser.add_argument_group('SCE options')
    sceGroup.add_argument('--weight-file', default=None, help="Weights for samples")
    sceGroup.add_argument('--maxIter', default=1000000, type=int, help="Maximum SCE iterations [default = 1000000]")
    sceGroup.add_argument('--nRepuSamp', default=5, type=int, help="Number of neighbours for calculating repulsion (1 or 5) [default = 5]")
    sceGroup.add_argument('--eta0', default=1, type=float, help="Learning rate [default = 1]")

    kmerGroup = parser.add_argument_group('Sequence input options')
    distType = kmerGroup.add_mutually_exclusive_group()
    distType.add_argument('--use-core', action='store_true', default=False, help="Use core distances")
    distType.add_argument('--use-accessory', action='store_true', default=False, help="Use accessory distances")
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
    if args.distances is None:
        sys.stderr.write("Calculating distances\n")
        if (args.alignment is not None):
            # alignment
            # TODO replace this with call to C++ program 
            sparse_matrix, consensus, seq_names = calculate_snp_matrix(fasta.file.name)
            P = calculate_distance_matrix(sparse_matrix, consensus, "dist", False)
        
        elif (args.accessory is not None):
            # accessory
            acc_mat = pd.read_csv(args.accessory, sep="\t", header=0, index_col=0, dtype=np.bool_)
            names = list(acc_mat.columns())
            P = pdist(acc_mat.values, 'hamming')

        elif (args.sequence is not None or args.sketches is not None):
            if args.min_k >= args.max_k or args.min_k < 9 or args.max_k > 31 or args.k_step < 2:
                sys.stderr.write("Minimum kmer size " + str(args.min_k) + " must be smaller than maximum kmer size " +
                                str(args.max_k) + "; range must be between 9 and 31, step must be at least one\n")
                sys.exit(1)
            kmers = np.arange(args.min_k, args.max_k + 1, args.k_step)

            if (not args.use_core and not args.use_accessory):
                sys.stderr.write("Must choose either --use-core or --use-accessory distances from sequence/sketches\n")
                sys.exit(1)
            elif (args.use_core):
                dist_col = 0
            elif (args.use_accessory):
                dist_col = 1
            
            if (args.sequence is not None):
                # sequence
                names, sequences = readRfile(args.sequence)
                P = pp_sketchlib.constructAndQuery(args.output, 
                                                        names, 
                                                        sequences, 
                                                        kmers, 
                                                        int(round(args.sketch_size/64)), 
                                                        args.min_count, 
                                                        args.cpus)[:,dist_col]

            elif (args.sketches is not None):
                # sketches
                names = getSeqsInDb(args.sketches + ".h5")
                kmers, sketch_size = readDBParams(args.sketches+ ".h5", kmers, int(round(args.sketch_size/64)))
                P = pp_sketchlib.queryDatabase(args.sketches, args.sketches, names, names, kmers, args.cpus)[:,dist_col]

        #***********************#
        #* Set up for SCE and  *#
        #* save distances      *#
        #***********************#
        if (len(names) < MIN_SAMPLES):
            sys.stderr.write("Less than minimum number of samples used (" + str(MIN_SAMPLES) + ")\n")
            sys.stderr.write("Distances calculated, but not running SCE\n")
        
        pd.Series(names).to_csv('names.txt', sep='\n', header=False, index=False)
        I, J = distVec(len(names))
        np.savez(args.output, I=I, J=J, P=P)

    # Load existing distances
    else:
        sys.stderr.write("Loading distances\n")
        npzfile = np.load(args.distances)
        I = npzfile['I']
        J = npzfile['J']
        P = npzfile['P']

    #***********************#
    #* run SCE             *#
    #***********************#
    sys.stderr.write("Running SCE\n")
    weights = np.ones((len(names)))
    if (args.weight_file):
        weights_in = pd.read_csv(args.weights, sep="\t", header=None, index_col=0)
        if (weights_in.index.symmetric_difference(names)):
            sys.stderr.write("Names in weights do not match sequences - using equal weights\n")
        else:
            intersecting_samples = weights_in.index.intersection(names)
            weights = weights_in.loc[intersecting_samples]
    
    embedding = np.array(wtsne(I, J, P, weights, args.maxIter, args.cpus, args.nRepuSamp, args.eta0))
    embedding = embedding.reshape(-1, 2)
    
    #***********************#
    #* plot embedding      *#
    #***********************#
    np.savetxt(args.output + ".embedding.txt", embedding)
    plotSCE(embedding, names, args.output)

    sys.exit(0)

if __name__ == "__main__":
    main()