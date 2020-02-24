/*
 ============================================================================
 Author      : Zhirong Yang
 Copyright   : Copyright by Zhirong Yang. All rights are reserved.
 Description : Stochastic Optimization for t-SNE
 ============================================================================
 */

// Modified by John Lees
#pragma once

#include <stdio.h>
#include <string.h>
#include <vector>
#include <iostream>

#ifndef DIM
#define DIM 2
#endif

#define MAX(a,b) ( (a) > (b) ? (a) : (b) )
#define MIN(a,b) ( (a) < (b) ? (a) : (b) )

template <class T>
std::vector<T> wtsne_init(const std::vector<long long>& I,
           const std::vector<long long>& J,
           std::vector<double>& P,
           std::vector<double>& weights)
{
    // Check input
    if (I.size() != J.size() || I.size() != P.size() || J.size() != P.size())
    {
        throw std::runtime_error("Mismatching sizes in input vectors");
    }
    if (I.size() < 2)
    {
        throw std::runtime_error("Input size too small");
    }
    long long nn = weights.size();
    long long ne = P.size();
    
    // Normalise distances and weights
    double Psum = 0.0;
    for (long long e=0; e<ne; e++) Psum += P[e];
    for (long long e=0; e<ne; e++) P[e] /= Psum;

    double weights_sum = 0.0;
    for (long long i=0; i<nn; i++) weights_sum += weights[i];
    for (long long i=0; i<nn; i++) weights[i] /= weights_sum;

    // Set starting Y0
    srand(0);
    std::vector<T> Y(nn*DIM);
    for (long long i = 0; i < nn; i++)
        for (long long d = 0; d < DIM; d++)
            Y[d + i*DIM] = rand() * 1e-4 / RAND_MAX;

    return Y;
}

std::vector<double> wtsne(std::vector<long long>& I,
           std::vector<long long>& J,
           std::vector<double>& P,
           std::vector<double>& weights,
           long long maxIter, 
           long long workerCount, 
           long long nRepuSamp,
           double eta0,
           bool bInit);

std::vector<float> wtsne_gpu(
	std::vector<long long>& I,
	std::vector<long long>& J,
	std::vector<double>& P,
	std::vector<double>& weights,
	long long maxIter, 
	int blockSize, 
	int blockCount,
	long long nRepuSamp,
	double eta0,
	bool bInit);