# Copyright 2019 John Lees

'''Main control function for pathoSCE'''

import os, sys

from .__init__ import __version__

from .dists import pairSnpDists, accessoryDists, sketchlibDists
from .sce import save_input, loadIJdist, runSCE, saveEmbedding
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
    mode.add_argument('--sketches',
                        default=None,
                        help='Work from sketch data')
    mode.add_argument('--distances',
                        default=None,
                        help='Work from pre-calculated distances')

    ioGroup = parser.add_argument_group('I/O options')
    ioGroup.add_argument('--output', default="pathoSCE", type=str, help='Prefix for output files [default = "pathoSCE"]')

    sceGroup = parser.add_argument_group('SCE options')
    sceGroup.add_argument('--no-preprocessing', default=False, action='store_true',
                                                     help="Turn off entropy pre-processing of distances")
    sceGroup.add_argument('--perplexity', default=15, type=float, help="Perplexity for distance to similarity "
                                                                            "conversion [default = 15]")
    sceGroup.add_argument('--weight-file', default=None, help="Weights for samples")
    sceGroup.add_argument('--maxIter', default=100000, type=int, help="Maximum SCE iterations [default = 100000]")
    sceGroup.add_argument('--nRepuSamp', default=5, type=int, help="Number of neighbours for calculating repulsion (1 or 5) [default = 5]")
    sceGroup.add_argument('--eta0', default=1, type=float, help="Learning rate [default = 1]")
    sceGroup.add_argument('--bInit', default=0, type=bool, help="1 for over-exaggeration in early stage [default = 0]")

    sketchGroup = parser.add_argument_group('Sketch options')
    sketchGroup.add_argument('--use-core', action='store_true', default=False, help="Use core distances")
    sketchGroup.add_argument('--use-accessory', action='store_true', default=False, help="Use accessory distances")
    sketchGroup.add_argument('--threshold', default=None, type=float, help='Maximum distance to consider [default = None]')
    sketchGroup.add_argument('--kNN', default=None, type=int, help='Number of k nearest neighbours to keep when sparsifying the distance matrix.')

    alnGroup = parser.add_argument_group('Alignment options')
    alnGroup.add_argument('--pairsnp-exe', default="pairsnp", type=str, help="Location of pairsnp executable (default='pairsnp')")

    other = parser.add_argument_group('Other')
    other.add_argument('--cpus',
                        type=int,
                        default=1,
                        help='Number of CPUs to use '
                             '[default = 1]')
    other.add_argument('--use-gpu', default=False, action='store_true',
                          help="Run SCE on the GPU. If this fails, the CPU will be used [default = False]")
    other.add_argument('--device-id',
                        type=int,
                        default=0,
                        help="GPU ID to use")
    other.add_argument('--blockSize',
                        type=int,
                        default=128,
                        help='CUDA blockSize '
                             '[default = 128]')
    other.add_argument('--blockCount',
                        type=int,
                        default=128,
                        help='CUDA blockCount '
                             '[default = 128]')
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
            I, J, dists, names = pairSnpDists(args.pairsnp_exe,
                                    args.alignment,
                                    args.output,
                                    args.threshold,
                                    args.kNN,
                                    args.cpus)
        elif (args.accessory is not None):
            I, J, dists, names = accessoryDists(args.accessory, args.sparse, args.kNN, args.threshold)

        elif (args.sketches is not None):
            # sketches
            if args.use_core:
                dist_col = 0
            elif args.use_accessory:
                dist_col = 1
            I, J, dists, names = sketchlibDists(args.sketches,
                                        dist_col,
                                        args.kNN,
                                        args.threshold,
                                        args.cpus,
                                        args.use_gpu,
                                        args.device_id)

        #***********************#
        #* Set up for SCE and  *#
        #* save distances      *#
        #***********************#
        save_input(I, J, dists, names, args.output)

    # Load existing distances
    else:
        sys.stderr.write("Loading distances\n")
        I, J, dists = loadIJdist(args.distances)

    #***********************#
    #* run SCE             *#
    #***********************#
    sys.stderr.write("Running SCE\n")
    SCE_opts = {'maxIter': args.maxIter,
               'cpus': args.cpus,
               'use_gpu': args.use_gpu,
               'device_id': args.device_id,
               'blockSize': args.blockSize,
               'blockCount': args.blockCount,
               'nRepuSamp': args.nRepuSamp,
               'eta0': args.eta0,
               'bInit': args.bInit}
    if args.no_preprocessing:
        SCE_opts['perplexity'] = -1
    else:
        SCE_opts['perplexity'] = args.perplexity

    embedding = runSCE(I, J, dists, args.weight_file, names, SCE_opts)
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
