Installation
============

Installing with conda (recommended)
-----------------------------------
.. important::
    Our conda package is currently pending. Until it is available
    please follow the manual installation information below.

If you do not have ``conda`` you can install it through
`miniconda <https://conda.io/miniconda.html>`_ and then add the necessary
channels::

    conda config --add channels defaults
    conda config --add channels bioconda
    conda config --add channels conda-forge

Then run::

    conda install mandrake

If you are having conflict issues with conda, our advice would be:

- Remove and reinstall miniconda.
- Never install anything in the base environment
- Create a new environment for mandrake with ``conda create -n mandrake_env poppunk``

conda-forge also has some helpful tips: https://conda-forge.org/docs/user/tipsandtricks.html

Installing manually
-------------------
You will need to install the dependencies, which are listed in ``environment.yml``.
We would still recommend using conda to do this::

    conda create -n mandrake_env python
    conda env update -n mandrake_env --file environment.yml
    conda activate mandrake_env

You can then install by running::

    python setup.py install

If you have the CUDA toolkit installed and ``nvcc`` on your path, this will also
compile the GPU code::

    -- CUDA found, compiling both GPU and CPU code

By default the code will be built for CUDA SM versions 70 (V100), 75 (20xx series), 80 (A100) and 86 (30xx series).
If you need more help on getting the GPU code to work, please see the page
in the `pp-sketchlib docs <https://poppunk.readthedocs.io/en/latest/gpu.html>`__, which
uses the same build procedure.

Developer notes
^^^^^^^^^^^^^^^
Install the debug build (which can be stepped through with ``gdb`` and ``cuda-gdb``)
by running::

    python setup.py install --debug

To run::

    cuda-gdb python
    set args mandrake-runner.py <args>
    r

To run without installing, run::

    python setup.py build

and add the following lines to the top of each python module file::

    import os, sys
    sys.path.insert(0, os.path.dirname(__file__) + '/../build/lib.macosx-10.9-x86_64-3.9')
    sys.path.insert(0, os.path.dirname(__file__) + '/../build/lib.linux-x86_64-3.9')

To change the compiler used, edit the following part of ``setup.py``::

    cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
              '-DPYTHON_EXECUTABLE=' + sys.executable,
              '-DCMAKE_C_COMPILER=gcc-10',
              '-DCMAKE_CXX_COMPILER=g++-10',
              '-DCMAKE_VERBOSE_MAKEFILE:BOOL=ON']

To profile the GPU code, uncomment the lines under 'Set these to profile' in
``CMakeLists.txt`` (there are three of these, two at the top, one in the CUDA section)
and reinstall. Run nsight-systems with::

    nsys profile -o sce_<hash> -c cudaProfilerApi --trace cuda,osrt,openmp mandrake <args>

nsight-compute with::

    ncu -c 10 -o sce_<hash> --set full --target-processes all mandrake <args>
