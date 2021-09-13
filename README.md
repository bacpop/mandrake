# mandrake
Fast visualisation of the population structure of pathogens using Stochastic Cluster Embedding

## Installation

You will need some dependencies, which you can install through `conda`:
```
conda create -n pathosce python numpy pandas scipy scikit-learn hdbscan pp-sketchlib cmake pybind11 openmp blas gsl gfortran-ng cudatoolkit
```

You can the clone this repository, and run:
```
python setup.py install
```

### GPU acceleration
If you have the ability to compile CUDA (e.g. `nvcc`) you should see a message:
```
CUDA found, compiling both GPU and CPU code
```
otherwise only the CPU version will be compiled:
```
CUDA not found, compiling CPU code only
```

## Usage
After installing, an example command would look like this:
```
pathoSCE --sketches sketchlib --output sketch --use-accessory --cpus 2 --nRepuSamp 1 --maxIter 200000
```
This would use a file `sketchlib.h5` created by [pp-sketchlib](https://github.com/johnlees/pp-sketchlib)
to calculate accessory distances. Output can be found in `sketch_SCE_result.html`.

To use the GPU accelerated code add the `--use-gpu` flag. If not available in the library, the
CPU code will be used instead.