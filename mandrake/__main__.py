# Copyright 2019 John Lees

'''Main control function for mandrake'''

import sys
import re
import pandas as pd

from .__init__ import __version__

from .dists import pairSnpDists, accessoryDists, sketchlibDists
from .sce import save_input, loadIJdist, runSCE, saveEmbedding
from .clustering import runHDBSCAN
from .plot import plotSCE_html, plotSCE_mpl, plotSCE_hex

def get_options():
    import argparse

    description = 'Visualisation of genomic distances in pathogen populations'
    parser = argparse.ArgumentParser(description=description,
                                     prog='mandrake')

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
    ioGroup.add_argument('--animate', default=False, action='store_true', help='Create an animation of the embedding process')
    ioGroup.add_argument('--labels', default=None, help='Sample labels for plotting (overrides DBSCAN clusters)')
    ioGroup.add_argument('--output', default="mandrake", type=str, help='Prefix for output files [default = "mandrake"]')

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

    parallelGroup = parser.add_argument_group('Parallelism options')
    parallelGroup.add_argument('--n-workers', default=1, type=int, help="Number of workers to use, sets max parallelism [default = 1]")
    parallelGroup.add_argument('--cpus', type=int, default=1, help='Number of CPUs to use [default = 1]')
    parallelGroup.add_argument('--use-gpu', default=False, action='store_true',
                               help="Run SCE on the GPU. If this fails, the CPU will be used [default = False]")
    parallelGroup.add_argument('--device-id', type=int, default=0, help="GPU ID to use")
    parallelGroup.add_argument('--blockSize', type=int, default=128, help='CUDA blockSize [default = 128]')

    sketchGroup = parser.add_argument_group('Sketch options')
    sketchGroup.add_argument('--use-core', action='store_true', default=False, help="Use core distances")
    sketchGroup.add_argument('--use-accessory', action='store_true', default=False, help="Use accessory distances")

    distGroup = parser.add_mutually_exclusive_group(required=True)
    distGroup.add_argument('--threshold', default=None, type=float, help='Maximum distance to consider [default = None]')
    distGroup.add_argument('--kNN', default=None, type=int, help='Number of k nearest neighbours to keep when sparsifying the distance matrix.')

    other = parser.add_argument_group('Other')
    other.add_argument('--seed', type=int, default=1, help='Seed for random number generation')
    other.add_argument('--fp', type=int, choices=[32, 64], default=64,
                        help='Floating point precision when using a GPU')
    other.add_argument('--version', action='version',
                       version='%(prog)s '+__version__)

    return parser.parse_args()

def main():
    args = get_options()

    # Set n_workers to a sensible default
    if not (isinstance(args.cpus, int) and (args.cpus > 0)):
        raise ValueError("Invalid value for cpus")
    if (not args.use_gpu and args.n_workers < args.cpus)\
       or (args.use_gpu and args.n_workers < args.blockSize):
        sys.stderr.write("Number of workers less than number of available threads, "
                         "increasing n_workers automatically\n")
        if args.use_gpu:
            args.n_workers = args.blockSize
        else:
            args.n_workers = args.cpus

    if args.kNN is not None:
        if not (isinstance(args.kNN, int) and (args.kNN > 0)):
            raise ValueError("Invalid value for kNN")
        args.threshold = 0
    elif args.threshold is not None:
        if not (isinstance(args.threshold, float) and (args.threshold > 0) and (args.threshold <= 1)):
            raise ValueError("Invalid value for threshold")
        args.kNN = 0

    #***********************#
    #* Run seq -> distance *#
    #***********************#
    if args.distances is None:
        sys.stderr.write("Calculating distances\n")

        if (args.alignment is not None):
            I, J, dists, names = pairSnpDists(args.alignment,
                                    args.threshold,
                                    args.kNN,
                                    args.cpus)
        elif (args.accessory is not None):
            I, J, dists, names = accessoryDists(args.accessory,
                                                args.kNN,
                                                args.threshold,
                                                args.cpus)
        elif (args.sketches is not None):
            # sketches
            dist_col = 0
            if args.use_accessory:
                dist_col = 1
            args.sketches = re.sub(r"\.h5$", "", args.sketches)
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
        I, J, dists, names = loadIJdist(args.distances)

    #***********************#
    #* run SCE             *#
    #***********************#
    sys.stderr.write("Running SCE\n")
    SCE_opts = {'maxIter': args.maxIter,
                'animate': args.animate,
                'cpus': args.cpus,
                'use_gpu': args.use_gpu,
                'device_id': args.device_id,
                'blockSize': args.blockSize,
                'n_workers': args.n_workers,
                'fp': args.fp,
                'nRepuSamp': args.nRepuSamp,
                'eta0': args.eta0,
                'bInit': args.bInit,
                'seed': args.seed}
    if args.no_preprocessing:
        SCE_opts['perplexity'] = -1
    else:
        SCE_opts['perplexity'] = args.perplexity

    embedding_results, embedding_array = runSCE(I, J, dists, args.weight_file, names, SCE_opts)
    saveEmbedding(embedding_array, args.output)

    #***********************#
    #* run HDBSCAN         *#
    #***********************#
    if args.labels == None:
        sys.stderr.write("Running clustering\n")
        cluster_labels = runHDBSCAN(embedding_array)
    else:
        label_file = pd.read_csv(args.labels, sep="\t", header=None, index_col=0)
        cluster_labels = list(label_file.loc[names][1].values)

    #***********************#
    #* plot embedding      *#
    #***********************#
    sys.stderr.write("Drawing plots\n")
    plotSCE_html(embedding_array, names, cluster_labels, args.output, not args.labels)
    plotSCE_hex(embedding_array, args.output)
    plotSCE_mpl(embedding_array, embedding_results, cluster_labels, args.output, not args.labels)

    sys.exit(0)

if __name__ == "__main__":
    main()
