Input options
=============

Any of the following modes also accept ``--labels``, which give the categories
to colours points by in the output plots. This should be a tab-separated file
with no header, the first column containing sample names, and the second column
containing the labels for each sample.

There are two additional parameters for the distances::

  --threshold THRESHOLD
                        Maximum distance to consider [default = None]
  --kNN KNN             Number of k nearest neighbours to keep when sparsifying the distance
                        matrix.

Rather than using a full (dense) matrix of all pairwise distances, mandrake uses
a sparse matrix, ignoring large distances. This uses significantly less memory without
affecting results.

``--kNN`` sets the number of distances to keep for each sample, which will be the
:math:`k` closest. Set :math:`k` to a number smaller than the number of samples.
Memory use grows linearly with :math:`k`. Setting :math:`k` too small will miss global
structure in the data.

``--threshold`` instead picks a maximum distance that is considered meaningful, and
larger distances will be removed from the input.

Multiple sequence alignment
---------------------------
Provide a multi-fasta alignment with ``--alignment``. Distances will be calculated
using a modified form of the `pairsnp <https://github.com/gtonkinhill/pairsnp>`__ algorithm,
and sparsified based on ``-kNN`` or ``--threshold``.

If you are trying to align large numbers of sequences (e.g. SARS-CoV-2), the reference-guided
mode of `MAFFT <https://mafft.cbrc.jp/alignment/software/>`__ may be helpful::

    mafft --6merpair --thread -1 --keeplength --addfragments filtered_SC2.fasta \
    nCoV-2019.reference.fasta > MA_filt_SC2.fasta

Sketch database (assemblies or reads)
-------------------------------------
Provide a `pp-sketchlib <https://github.com/johnlees/pp-sketchlib>`__ database
with ``--sketches``, to calculate core and accessory distances
between the sketches. Core distances are used by default, but add ``--use-accessory`` to
alter this behaviour.

This should be a HDF5 file with suffix ``.h5`` produced by sketchlib, for example
by a command such as::

    sketchlib sketch -l sample_rfile.txt -o sketch_db --cpus 16

Gene presence/absence
---------------------
Pan-genome programs such as `roary <https://sanger-pathogens.github.io/Roary/>`__ and
`panaroo <https://gtonkinhill.github.io/panaroo/#/>`__ output a ``gene_presence_absence.Rtab``
file, which can be used with ``--accessory`` to calculate accessory distances (Hamming distances).

unitig counting programs such as `unitig-caller <https://github.com/johnlees/unitig-caller>`__
also output this file format, though the interpretation of the distances is slightly different
it can also be used as input.

Precalculated distances
-----------------------
After calculating distances, mandrake will save these as ``<output_prefix>.npz``. These
can be used as input without the need to compute them again with ``--distances``,
which is useful when you wish to run the embedding algorithm on the same data with
different parameters.
