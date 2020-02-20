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

std::vector<double> wtsne_init(const std::vector<long long>& I,
           const std::vector<long long>& J,
           std::vector<double>& P,
           std::vector<double>& weights);

std::vector<double> wtsne(std::vector<long long>& I,
           std::vector<long long>& J,
           std::vector<double>& P,
           std::vector<double>& weights,
           long long maxIter, 
           long long workerCount, 
           long long nRepuSamp,
           double eta0,
           bool bInit);

std::vector<double> wtsne_gpu(
	std::vector<long long>& I,
	std::vector<long long>& J,
	std::vector<double>& P,
	std::vector<double>& weights,
	long long maxIter, 
	long long workerCount,
	int blockSize, 
	int blockCount,
	long long nRepuSamp,
	double eta0,
	bool bInit);