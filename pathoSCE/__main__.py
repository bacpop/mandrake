# Copyright 2019 John Lees

'''Main control function for pathoSCE'''

import os, sys

import numpy as np
import pandas as pd

from .__init__ import __version__

from .dists import pairSnpDists, accessoryDists, sketchlibDists, sketchlibDbDists
from .sce import generateIJP, loadIJP, runSCE, saveEmbedding
from .sce import DEFAULT_THRESHOLD
from .clustering import runHDBSCAN
from .plot import plotSCE

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
    
    ioGroup = parser.add_argument_group('I/O options')
    ioGroup.add_argument('--output', default="pathoSCE", type=str, help='Prefix for output files [default = "pathoSCE"]')

    distanceGroup = parser.add_argument_group('Distance options')
    distanceGroup.add_argument('--no-preprocessing', default=False, action='store_true',
                                                     help="Turn of entropy pre-processing of distances")
    distanceGroup.add_argument('--perplexity', default=15, type=float, help="Perplexity for distance to similarity "
                                                                            "conversion [default = 15]")
    distanceGroup.add_argument('--sparse', default=False, action='store_true', 
                               help='Use sparse matrix calculations to speed up'
                                    'distance calculation from --accessory [default = False]')
    distanceGroup.add_argument('--threshold', default=DEFAULT_THRESHOLD, type=float, help='Maximum distance to consider [default = 0]')
    distanceGroup.add_argument('--kNN', default=None, type=int, help='Number of k nearest neighbours to keep when sparsifying the distance matrix.')

    sceGroup = parser.add_argument_group('SCE options')
    sceGroup.add_argument('--use-gpu', default=False, action='store_true',
                          help="Run SCE on the GPU. If this fails, the CPU will be used [default = False]")
    sceGroup.add_argument('--weight-file', default=None, help="Weights for samples")
    sceGroup.add_argument('--maxIter', default=100000, type=int, help="Maximum SCE iterations [default = 100000]")
    sceGroup.add_argument('--nRepuSamp', default=5, type=int, help="Number of neighbours for calculating repulsion (1 or 5) [default = 5]")
    sceGroup.add_argument('--eta0', default=1, type=float, help="Learning rate [default = 1]")
    sceGroup.add_argument('--bInit', default=0, type=bool, help="1 for over-exaggeration in early stage [default = 0]")

    kmerGroup = parser.add_argument_group('Sequence input options')
    distType = kmerGroup.add_mutually_exclusive_group()
    distType.add_argument('--use-core', action='store_true', default=False, help="Use core distances")
    distType.add_argument('--use-accessory', action='store_true', default=False, help="Use accessory distances")
    kmerGroup.add_argument('--min-k', default = 13, type=int, help='Minimum kmer length [default = 13]')
    kmerGroup.add_argument('--max-k', default = 29, type=int, help='Maximum kmer length [default = 29]')
    kmerGroup.add_argument('--k-step', default = 4, type=int, help='K-mer step size [default = 4]')
    kmerGroup.add_argument('--sketch-size', default=10000, type=int, help='Kmer sketch size [default = 10000]')
    kmerGroup.add_argument('--min-count', default=20, type=int, help='Minimum k-mer count from reads [default = 20]')

    alnGroup = parser.add_argument_group('Alignment options')
    alnGroup.add_argument('--pairsnp-exe', default="pairsnp", type=str, help="Location of pairsnp executable (default='pairsnp')")

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
            P, names = pairSnpDists(args.pairsnp_exe, 
                                    args.alignment, 
                                    args.output, 
                                    args.threshold, 
                                    args.kNN, 
                                    args.cpus)
        elif (args.accessory is not None):
            P, names = accessoryDists(args.accessory, args.sparse, args.kNN)
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
                P, names = sketchlibDists(args.sequence, 
                                          args.output, 
                                          kmers, 
                                          args.sketch_size, 
                                          args.min_count, 
                                          dist_col,
                                          args.kNN,
                                          args.threshold,
                                          args.cpus)

            elif (args.sketches is not None):
                # sketches
                P, names = sketchlibDbDists(args.sketches, 
                                            kmers, 
                                            args.sketch_size, 
                                            dist_col,
                                            args.kNN,
                                            args.threshold,
                                            args.cpus)

        #***********************#
        #* Set up for SCE and  *#
        #* save distances      *#
        #***********************#
        I, J, P = generateIJP(names, 
                              args.output, 
                              P, 
                              not args.no_preprocessing, 
                              args.perplexity)

    # Load existing distances
    else:
        sys.stderr.write("Loading distances\n")
        I, J, P = loadIJP(args.distances)

    #***********************#
    #* run SCE             *#
    #***********************#
    sys.stderr.write("Running SCE\n")
    SCE_opts = {'maxIter': args.maxIter,
               'cpus': args.cpus,
               'nRepuSamp': args.nRepuSamp,
               'eta0': args.eta0,
               'bInit': args.bInit}
    embedding = runSCE(I, J, P, args.weight_file, names, SCE_opts, args.use_gpu)
    saveEmbedding(embedding, args.output)

    #***********************#
    #* run HDBSCAN         *#
    #***********************#
    hdb_clusters = runHDBSCAN(embedding)
    
    #***********************#
    #* plot embedding      *#
    #***********************#
    plotSCE(embedding, names, hdb_clusters, args.output)

    sys.exit(0)

if __name__ == "__main__":
    main()