nvcc wtsne_gpu.cu -o  wtsne_gpu --cudart static -O3 --relocatable-device-code=false -gencode arch=compute_35,code=compute_35 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_50,code=sm_50 -lgsl -lgslcblas

