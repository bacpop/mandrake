#!/usr/bin/env python
# Copyright 2021 John Lees and Gerry Tonkin-Hill

"""Tests for PopPUNK"""

import subprocess
import os
import sys

import SCE
print(SCE.version)

sys.stderr.write("Extracting example datasets\n")
if not os.path.isfile("listeria.h5"):
    subprocess.run("bzip2 -d -c data/listeria.h5.bz2 > listeria.h5", shell=True, check=True)
if not os.path.isfile("sub5k_hiv_refs_prrt_trim.fas"):
    subprocess.run("bzip2 -d -c data/sub5k_hiv_refs_prrt_trim.fas.bz2 > sub5k_hiv_refs_prrt_trim.fas", shell=True, check=True)
if not os.path.isfile("gene_presence_absence.Rtab"):
    subprocess.run("bzip2 -d -c data/gene_presence_absence.Rtab.bz2 > gene_presence_absence.Rtab", shell=True, check=True)

if os.environ.get("MANDRAKE_PYTHON"):
    python_cmd = os.environ.get("MANDRAKE_PYTHON")
else:
    python_cmd = "python"

subprocess.run(python_cmd + " ../mandrake-runner.py --version", shell=True, check=True)

sys.stderr.write("Running with each input type\n")
subprocess.run(python_cmd + " ../mandrake-runner.py --alignment sub5k_hiv_refs_prrt_trim.fas --kNN 50 --cpus 2 --maxIter 10000000", shell=True, check=True)
subprocess.run(python_cmd + " ../mandrake-runner.py --sketches listeria.h5 --threshold 0.02 --use-accessory --cpus 2 --maxIter 10000000", shell=True, check=True)
subprocess.run(python_cmd + " ../mandrake-runner.py --accessory gene_presence_absence.Rtab --kNN 50 --cpus 2 --maxIter 10000000", shell=True, check=True)

sys.stderr.write("Re-running with different SCE options\n")
subprocess.run(python_cmd + " ../mandrake-runner.py --sketches listeria.h5 --kNN 50 --cpus 2 --maxIter 10000000", shell=True, check=True)
subprocess.run(python_cmd + " ../mandrake-runner.py --distances mandrake.npz --maxIter 10000000 --eta0 2 --bInit 1 --perplexity 5", shell=True, check=True)
subprocess.run(python_cmd + " ../mandrake-runner.py --distances mandrake.npz --maxIter 10000000 --animate --output animation", shell=True, check=True)
subprocess.run(python_cmd + " ../mandrake-runner.py --distances mandrake.npz --maxIter 10000000 --labels data/labels.txt", shell=True, check=True)
subprocess.run(python_cmd + " ../mandrake-runner.py --distances mandrake.npz --maxIter 10000000 --weight-file data/weights.txt", shell=True, check=True)

# test updating order is correct
sys.stderr.write("Processing can be turned off\n")
# This won't necessarily work
# subprocess.run(python_cmd + " ../mandrake-runner.py --distances mandrake.npz --maxIter 10000000 --no-preprocessing", shell=True, check=True)
subprocess.run(python_cmd + " ../mandrake-runner.py --distances mandrake.npz --maxIter 10000000 --no-clustering", shell=True, check=True)

sys.stderr.write("Tests completed\n")
