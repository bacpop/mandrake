g++ -std=c++11 wtsne_cpu.cpp -o wtsne_cpu -lm -lgsl -lgslcblas -lgomp -fopenmp -Ofast -march=native -ffast-math
