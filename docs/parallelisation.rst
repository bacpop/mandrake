Parallelisation
================

To increase the speed of mandrake, you can alter the following options::

  --n-workers N_WORKERS
                        Number of workers to use, sets max parallelism [default = 1]
  --cpus CPUS           Number of CPUs to use [default = 1]
  --use-gpu             Run SCE on the GPU. If this fails, the CPU will be used [default =
                        False]
  --device-id DEVICE_ID
                        GPU ID to use
  --blockSize BLOCKSIZE
                        CUDA blockSize [default = 128]

Set the number of cores to use by specifying ``--cpus``. This will set the parallelism
for the SCE (embedding), but also carries through to the distance calculation.

To use a GPU you will need:
- A CUDA enabled GPU available with appropriate drivers.
- A version of the code compiled with CUDA.
- To add the ``--use-gpu`` flag.

A GPU will also be used for sketch distances, if ``pp-sketchlib`` is also set up
to use GPUs. You can choose a GPU with the ``--device-id`` flag, or the ``CUDA_VISIBLE_DEVICES``
environment variable (in which case keep device ID as the default; this is how
some HPCs set the GPU).

You can change the block size for the SCE process with ``--blockSize`` (this
should be a multiple of 32, and likely one of 32, 64, 128 or 256). The default is
128, and you probably don't need to change this unless you are interested in CUDA.

Setting the number of workers
-----------------------------
The number of workers ``--n-workers`` sets the maximum amount of parallelism
possible. Each worker will randomly pick a sample pair to update the attraction
between, and then ``--nRepuSamp`` pairs to update the repulsion between. These
can be run in parallel at each iteration.

If you are running with CPU cores, you probably want to set the number of workers
equal to the number of available cores (this will be done automatically). If you are
running on a GPU you should set at least as many workers as the block size. However,
GPUs become more efficient with more threads, up to about 100k threads. So you want to set
as many workers as possible, ideally up to 100k (and probably an integer multiple of the block size).

However, the more workers running in parallel there are relative to the number
of samples, the more likely two or more workers will try and update the same
sample at the same time. The frequency of these attempted overwrites is shown by
the percentage of clashes in the progress bar.

The CPU code largely avoids multiple synchronous worker updates, but you may find
that speedup decreases as clash rate increases. The GPU code does not correct for clashes
(but does maintain memory validity), instead relying on the stochastic nature of
the algorithm to give a correct result. Speed is likely to decreases with clashes,
but at large values you may find issues with convergence. If the clash rate is
approaching 100%, you should probably decrease the number of workers, even if this
takes you below the optimum for a GPU.
