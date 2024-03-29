SCE parameters
==============

The options for the SCE algorithm are explained below, other options are
explained in :doc:`input`, :doc:`animation` and :doc:`parallelisation`.

The progress bar shows the percentage complete, the current learning rate :math:`\eta`,
the divergence :math:`Eq` (and clash rate, see :doc:`parallelisation`).::

    Optimizing	 Progress: 99.9%, eta=0.0010, Eq=0.0547636151, clashes=0.1%
    Optimizing done in 20s

The algorithm should be run until :math:`Eq` stabilises.

SCE options::

  --perplexity PERPLEXITY
                        Perplexity for distance to similarity conversion [default = 15]
  --no-preprocessing    Turn off entropy pre-processing of distances
  --weight-file WEIGHT_FILE
                        Weights for samples
  --maxIter MAXITER     Maximum SCE iterations [default = 100000]
  --nRepuSamp NREPUSAMP
                        Number of neighbours for calculating repulsion (1 or 5) [default = 5]
  --eta0 ETA0           Learning rate [default = 1]
  --bInit BINIT         1 for over-exaggeration in early stage [default = 0]
  --no-clustering       Turn off HDBSCAN clustering after SCE

- ``--perplexity`` roughly sets the balance between global and local structure, smaller
  values making fewer neighbours matter, and emphasising local structure. Typical
  values are between 5 and 50, but making plots at multiple values is often useful.
- ``--no-preprocessing`` uses raw similarities as input, rather than a probability
  distribtion using a desired perplexity. Do not use this option unless you are having
  issues setting the perplexity.
- ``--weight-file`` allows you to give different samples different weights in
  being picked to attract or repel. The default is to equally weight samples, but
  you could for example weight by cluster size or its inverse. This requires a tab-separated
  file with no header, the first column with sample names, the second column with their weights.
- ``--maxIter`` sets how long the algorithm will run for. Larger datasets will need more
  iterations. We recommend running until :math:`Eq` stabilises.
- ``--nRepuSamp`` is the number of neighbours used for calculating the repulsion at
  each iteration. This can be one or five.
- ``--eta0`` sets the scale of the learning rate. A higher value will move points
  around more at each iteration.
- ``--bInit`` turns on over-exaggeration, which sets the stength of attraction four times
  higher in the first 10% of the iterations.
- ``--no-clustering`` turns off the HDBSCAN clustering if you don't want spatial clustering
  run on the embedding (which may stall when no structure has been found). This is
  implied if ``--labels`` have been provided.
