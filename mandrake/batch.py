#!/usr/bin/env python
# vim: set fileencoding=<utf-8> :
# Copyright 2021 John Lees

# universal
import sys
import re

import pp_sketchlib
from poppunk_refine import extend

from .sketchlib import get_kmer_sizes, get_seqs_in_db
from .sce import save_input

epsilon = 1e-10

def get_options():
    import argparse

    parser = argparse.ArgumentParser(description='Batch distances from sketches',
                                     prog='mandrake_batch_dists')

    # input options
    ioGroup = parser.add_argument_group('Input and output file options')
    ioGroup.add_argument('--db', help='HDF5 sketch database', required=True)
    ioGroup.add_argument('--n-batches', help='Number of batches for process if --batch-file is not specified',
                         type=int, required=True)
    ioGroup.add_argument(
        '--output', help='Prefix for output files', required=True)

    # analysis options
    aGroup = parser.add_argument_group('Distance options')
    aGroup.add_argument(
        '--kNN', help='Number of nearest neighbours to keep', type=int, required=True)
    aGroup.add_argument('--adj-random', help='Use random adjustments',
                        default=False, action='store_true')
    aGroup.add_argument('--use-accessory', action='store_true',
                        default=False, help="Use accessory distances [default = use core]")

    pGroup = parser.add_argument_group('Parallelism options')
    pGroup.add_argument('--cpus', help='Number of threads for parallelisation (int)',
                        type=int,
                        default=1)
    pGroup.add_argument('--use-gpu', help='Use GPU for distance calculations',
                        default=False,
                        action='store_true')
    pGroup.add_argument('--device-id', help='GPU device ID (int)',
                        type=int,
                        default=0)

    return parser.parse_args()


def main():
    args = get_options()
    sketch_db = re.sub(r"\.h5$", "", args.db)

    names = get_seqs_in_db(sketch_db + ".h5")
    batches = []
    names_per_batch = len(names) // args.n_batches
    n_big_batches = len(names) % args.n_batches
    if names_per_batch < args.kNN:
        raise ValueError("kNN must be smaller than the samples per batch")

    start = 0
    for batch_idx in range(args.n_batches):
        if batch_idx < n_big_batches:
            end = start + names_per_batch + 1
        else:
            end = start + names_per_batch
        batches.append(names[start:end])
        start = end

    kmers = get_kmer_sizes(sketch_db + ".h5")
    if (len(kmers) == 1):
        jaccard = True
    else:
        jaccard = False
    dist_col = 0
    if args.use_accessory and not jaccard:
        dist_col = 1

    sys.stderr.write("Batch 1 of " + str(len(batches)) + "\n")
    ref_names = batches[0]
    rrDense = pp_sketchlib.queryDatabase(
        sketch_db, sketch_db, ref_names, ref_names, kmers, args.adj_random,
        jaccard, args.cpus, args.use_gpu, args.device_id
    )
    if jaccard:
        rrDense = 1 - rrDense

    row, col, data = \
        pp_sketchlib.sparsifyDists(
            pp_sketchlib.longToSquare(rrDense[:, [dist_col]], args.cpus),
            0,
            args.kNN)

    for batch_idx, batch in enumerate(batches[1:]):
        sys.stderr.write("Batch " + str(batch_idx + 2) + " of " + str(len(batches)) + "\n")
        qqDense = pp_sketchlib.queryDatabase(
          sketch_db, sketch_db, batch, batch, kmers, args.adj_random,
          jaccard, args.cpus, args.use_gpu, args.device_id
        )
        if jaccard:
            qqDense = 1 - qqDense

        qqSquare = pp_sketchlib.longToSquare(qqDense[:, [dist_col]], args.cpus)
        qqSquare[qqSquare < epsilon] = epsilon

        qrDense = pp_sketchlib.queryDatabase(
          sketch_db, sketch_db, ref_names, batch, kmers, args.adj_random,
          jaccard, args.cpus, args.use_gpu, args.device_id
        )
        if jaccard:
            qrDense = 1 - qrDense

        n_ref = len(ref_names)
        n_query = len(batch)
        qrRect = qrDense[:, [dist_col]].reshape(n_query, n_ref).T
        qrRect[qrRect < epsilon] = epsilon

        row, col, data = \
            extend((row, col, data), qqSquare, qrRect, args.kNN, args.cpus)

        ref_names += batch

    save_input(row, col, data, names, args.output)


if __name__ == "__main__":
    main()
