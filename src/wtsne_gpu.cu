/*
 ============================================================================
 Author      : Zhirong Yang
 Copyright   : Copyright by Zhirong Yang. All rights are reserved.
 Description : Weighted t-SNE with stochastic optimization
 ============================================================================
 */

#include <numeric>
#include <stdlib.h>
#include <stdio.h>
#include <cfloat>
#include <curand_kernel.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

#include "wtsne.hpp"

static void
CheckCudaErrorAux(const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

/**
 * Check the return value of the CUDA runtime API call and exit
 * the application if the call has failed.
 */
 static void CheckCudaErrorAux(const char *file, unsigned line,
	const char *statement, cudaError_t err) 
{
	if (err == cudaSuccess)
		return;
	std::cerr << statement << " returned " << cudaGetErrorString(err) << "("
			<< err << ") at " << file << ":" << line << std::endl;
	exit(1);
}

/****************************
* Functions to move data    *
* and functions on/off GPUs *
****************************/

// Moves arrays onto GPU
void allocateDataAndCopy2Device(float* d_Y, float* d_I, float* d_J, float* d_Eq,
								std::vector<double>& Y,
								std::vector<double>& I,
								std::vector<double>& J,
								long long nn, long long ne,
								float* d_qsum, int* d_qcount, 
								float* d_qsum_total, float* d_qcount_total,
								int nWorker) {
	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_Y, sizeof(float)*nn*DIM));
	CUDA_CHECK_RETURN(cudaMalloc((void** )&d_I, sizeof(long long) * ne));
	CUDA_CHECK_RETURN(cudaMalloc((void** )&d_J, sizeof(long long) * ne));

	CUDA_CHECK_RETURN(
			cudaMemcpy(d_Y, Y.data(), sizeof(float)*nn*DIM, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(
			cudaMemcpy(d_I, I.data(), sizeof(long long) * ne, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(
			cudaMemcpy(d_J, J.data(), sizeof(long long) * ne, cudaMemcpyHostToDevice));

	float Eq = 1;
	CUDA_CHECK_RETURN(cudaMalloc((void** )&d_Eq, sizeof(float)));
	CUDA_CHECK_RETURN(
			cudaMemcpy(d_Eq, &Eq, sizeof(float), cudaMemcpyHostToDevice));

	CUDA_CHECK_RETURN(cudaMalloc((void ** )&d_qsum, nWorker * sizeof(float)));
	CUDA_CHECK_RETURN(
			cudaMalloc((void ** )&d_qcount, nWorker * sizeof(long long)));
	CUDA_CHECK_RETURN(cudaMalloc((void ** )&d_qsum_total, sizeof(float)));
	CUDA_CHECK_RETURN(cudaMalloc((void ** )&d_qcount_total, sizeof(long long)));
}

void setupDiscreteDistribution(curandState *d_nnStates1, curandState *d_nnStates2, curandState *d_neStates,
	std::vector<double>& P, std::vector<double>& weights,
	gsl_ran_discrete_t *d_gsl_de, gsl_ran_discrete_t *d_gsl_dn,
	double *d_gsl_de_F, double *d_gsl_dn_F,
	size_t *d_gsl_de_A, size_t *d_gsl_dn_A,
	int blockCount, int blockSize, long long nn, long long ne)
{
	CUDA_CHECK_RETURN(
	cudaMalloc((void ** )&d_nnStates1,
	blockCount * blockSize * sizeof(curandState)));
	CUDA_CHECK_RETURN(
	cudaMalloc((void ** )&d_nnStates2,
	blockCount * blockSize * sizeof(curandState)));
	CUDA_CHECK_RETURN(
	cudaMalloc((void ** )&d_neStates,
	blockCount * blockSize * sizeof(curandState)));
	setupCURANDKernel<<<blockCount, blockSize>>>(d_nnStates1, d_nnStates2,
	d_neStates);

	// These are free'd at the end of the function
	gsl_rng_env_setup();
	gsl_ran_discrete_t * gsl_de = gsl_ran_discrete_preproc(ne, P.data());
	gsl_ran_discrete_t * gsl_dn = gsl_ran_discrete_preproc(nn, weights.data());

	CUDA_CHECK_RETURN(
	cudaMalloc((void ** )&d_gsl_de, sizeof(gsl_ran_discrete_t)));
	CUDA_CHECK_RETURN(
	cudaMalloc((void ** )&d_gsl_dn, sizeof(gsl_ran_discrete_t)));
	CUDA_CHECK_RETURN(cudaMalloc((void ** )&d_gsl_de_A, sizeof(size_t) * ne));
	CUDA_CHECK_RETURN(cudaMalloc((void ** )&d_gsl_de_F, sizeof(double) * ne));
	CUDA_CHECK_RETURN(cudaMalloc((void ** )&d_gsl_dn_A, sizeof(size_t) * nn));
	CUDA_CHECK_RETURN(cudaMalloc((void ** )&d_gsl_dn_F, sizeof(double) * nn));
	CUDA_CHECK_RETURN(
	cudaMemcpy(d_gsl_de, gsl_de, sizeof(gsl_ran_discrete_t),
	cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(
	cudaMemcpy(d_gsl_de_A, gsl_de->A, sizeof(size_t) * ne,
	cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(
	cudaMemcpy(d_gsl_de_F, gsl_de->F, sizeof(double) * ne,
	cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(
	cudaMemcpy(d_gsl_dn, gsl_dn, sizeof(gsl_ran_discrete_t),
	cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(
	cudaMemcpy(d_gsl_dn_A, gsl_dn->A, sizeof(size_t) * nn,
	cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(
	cudaMemcpy(d_gsl_dn_F, gsl_dn->F, sizeof(double) * nn,
	cudaMemcpyHostToDevice));
	assembleGSLKernel<<<1, 1>>>(d_gsl_de, d_gsl_de_A, d_gsl_de_F, d_gsl_dn,
	d_gsl_dn_A, d_gsl_dn_F);
	gsl_ran_discrete_free(gsl_de);
	gsl_ran_discrete_free(gsl_dn);
}

// Frees memory on GPU
void freeDataInDevice(float* d_Y, float* d_I, float* d_J,
	float* d_qsum, int* d_qcount, 
	float* d_qsum_total, float* d_qcount_total,
	curandState *d_nnStates1, curandState *d_nnStates2, curandState *d_neStates,
	gsl_ran_discrete_t *d_gsl_de, gsl_ran_discrete_t *d_gsl_dn,
	double *d_gsl_de_F, double *d_gsl_dn_F,
	size_t *d_gsl_de_A, size_t *d_gsl_dn_A)
{
	// data
	CUDA_CHECK_RETURN(cudaFree(d_Y));
	CUDA_CHECK_RETURN(cudaFree(d_I));
	CUDA_CHECK_RETURN(cudaFree(d_J));
	CUDA_CHECK_RETURN(cudaFree(d_qsum));
	CUDA_CHECK_RETURN(cudaFree(d_qcount));
	CUDA_CHECK_RETURN(cudaFree(d_qsum_total));
	CUDA_CHECK_RETURN(cudaFree(d_qcount_total));

	// rng
	CUDA_CHECK_RETURN(cudaFree(d_nnStates1));
	CUDA_CHECK_RETURN(cudaFree(d_nnStates2));
	CUDA_CHECK_RETURN(cudaFree(d_neStates));
	CUDA_CHECK_RETURN(cudaFree(d_gsl_de_A));
	CUDA_CHECK_RETURN(cudaFree(d_gsl_de_F));
	CUDA_CHECK_RETURN(cudaFree(d_gsl_de));
	CUDA_CHECK_RETURN(cudaFree(d_gsl_dn_A));
	CUDA_CHECK_RETURN(cudaFree(d_gsl_dn_F));
	CUDA_CHECK_RETURN(cudaFree(d_gsl_dn));
}

/****************************
* Functions run on the      *
* device                    *
****************************/
__device__ size_t my_curand_discrete(curandState *state,
		const gsl_ran_discrete_t *g) {
	size_t c = 0;
	double u, f;
	u = curand_uniform(state);
	c = (u * (g->K));
	f = (g->F)[c];
	if (f == 1.0)
		return c;

	if (u < f) {
		return c;
	} else {
		return (g->A)[c];
	}
}

__global__ void setupCURANDKernel(curandState *nnStates1,
		curandState *nnStates2, curandState *neStates) {
	long long workerIdx = (long long) (blockIdx.x * blockDim.x + threadIdx.x);
	curand_init(314159, /* the seed */
	workerIdx, /* the sequence number */
	0, /* not use the offset */
	&nnStates1[workerIdx]);
	curand_init(314159 + 1, /* the seed */
	workerIdx, /* the sequence number */
	0, /* not use the offset */
	&nnStates2[workerIdx]);
	curand_init(271828, /* the seed */
	workerIdx, /* the sequence number */
	0, /* not use the offset */
	&neStates[workerIdx]);
}

__global__ void assembleGSLKernel(gsl_ran_discrete_t *d_gsl_de,
		size_t *d_gsl_de_A, double *d_gsl_de_F, gsl_ran_discrete_t *d_gsl_dn,
		size_t *d_gsl_dn_A, double *d_gsl_dn_F) {
	d_gsl_de->A = d_gsl_de_A;
	d_gsl_de->F = d_gsl_de_F;
	d_gsl_dn->A = d_gsl_dn_A;
	d_gsl_dn->F = d_gsl_dn_F;
}

// Updates the embedding
// (These arguments are ok as they are passed the array pointers d_Y, d_I etc
// which are already on the device; NOT the redefined Y vectors etc)
__global__ void wtsneUpdateYKernel(curandState *nnStates1,
		curandState *nnStates2, curandState *neStates,
		gsl_ran_discrete_t* d_gsl_dn, gsl_ran_discrete_t* d_gsl_de, float *Y,
		long long *I, long long *J, float *d_Eq, float *qsum, int *qcount, long long nn,
		long long ne, float eta, long long nRepuSamp, float nsq, float attrCoef) {
	int workerIdx = blockIdx.x * blockDim.x + threadIdx.x;
	float dY[DIM];
	float c = 1.0 / ((*d_Eq) * nsq);
	qsum[workerIdx] = 0.0;
	qcount[workerIdx] = 0;

	float repuCoef = 2 * c / nRepuSamp * nsq;
	for (long long r = 0; r < nRepuSamp + 1; r++) {
		long long k, l;
		if (r == 0) {
			long long e = (long long) (my_curand_discrete(neStates + workerIdx,
					d_gsl_de) % ne);
			k = I[e];
			l = J[e];
		} else {
			k = (long long) (my_curand_discrete(nnStates1 + workerIdx, d_gsl_dn)
					% nn);
			l = (long long) (my_curand_discrete(nnStates2 + workerIdx, d_gsl_dn)
					% nn);
		}

		if (k == l)
			continue;

		long long lk = k * DIM;
		long long ll = l * DIM;
		float dist2 = 0.0;
		for (long long d = 0; d < DIM; d++) {
			dY[d] = Y[d + lk] - Y[d + ll];
			dist2 += dY[d] * dY[d];
		}
		float q = 1.0 / (1 + dist2);

		float g;
		if (r == 0)
			g = -attrCoef * q;
		else
			g = repuCoef * q * q;

		for (long long d = 0; d < DIM; d++) {
			float gain = eta * g * dY[d];
			Y[d + lk] += gain;
			Y[d + ll] -= gain;

		}
		qsum[workerIdx] += q;
		qcount[workerIdx]++;
	}
}

__global__ void resetQsumQCountTotalKernel(float *d_qsum_total,
		int *d_qcount_total) {
	(*d_qsum_total) = 0.0;
	(*d_qcount_total) = 0;
}

template<typename T>
__global__ void reduceSumArrayKernel(T *array, int n, T* arraySum) {
	T sum = 0;
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
			i += blockDim.x * gridDim.x) {
		sum += array[i];
	}
	atomicAdd(arraySum, sum);
}

__global__ void updateEqKernel(float *d_Eq, float *d_qsum_total,
		int* d_qcount_total, float nsq) {
	(*d_Eq) = ((*d_Eq) * nsq + (*d_qsum_total)) / (nsq + (*d_qcount_total));
}

/****************************
* Main control function     *
****************************/
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
	bool bInit) 
{
	// Check input
	Y = wtsne_init(I, J, P, weights);
    long long nn = weights.size();
    long long ne = P.size();
	
	// Initialise CUDA
	cudaSetDevice(0);
	cudaDeviceReset();
	nWorker = blockSize * blockCount;
	float nsq = (float) nn * (nn - 1);

	// Create pointers for mallocs
	long long *d_I, *d_J;
	float *d_Y;
	float *d_Eq;
	float *d_qsum, *d_qsum_total;
	int *d_qcount, *d_qcount_total;

	// malloc on device
	allocateDataAndCopy2Device(d_Y, d_I, d_J, d_Eq,
							   Y, I, J, nn, ne,
							   d_qsum, d_qcount,
							   d_qsum_total, d_qcount_total,
							   nWorker);

	// Set up random number generation
	curandState *d_nnStates1, *d_nnStates2;
	curandState *d_neStates;
	gsl_ran_discrete_t *d_gsl_de, *d_gsl_dn;
	double *d_gsl_de_F, *d_gsl_dn_F;
	size_t *d_gsl_de_A, *d_gsl_dn_A;
	setupDiscreteDistribution(d_nnStates1, d_nnStates2, d_neStates,
							  P, weights,
		                      *d_gsl_de, *d_gsl_dn,
		                      *d_gsl_de_F, *d_gsl_dn_F,
							  *d_gsl_de_A, *d_gsl_dn_A,
							  blockCount, blockSize, 
							  nn, ne);

	// Main SCE loop
	for (long long iter = 0; iter < maxIter; iter++) {
		float eta = eta0 * (1 - (float) iter / (maxIter - 1));
		eta = MAX(eta, eta0 * 1e-4);

		float attrCoef = (bInit && iter < maxIter / 10) ? 8 : 2;
		wtsneUpdateYKernel<<<blockCount, blockSize>>>(d_nnStates1, d_nnStates2,
				d_neStates, d_gsl_dn, d_gsl_de, d_Y, d_I, d_J, d_Eq, d_qsum,
				d_qcount, nn, ne, eta, nRepuSamp, nsq, attrCoef);

		resetQsumQCountTotalKernel<<<1, 1>>>(d_qsum_total, d_qcount_total);
		reduceSumArrayKernel<<<16, 128>>>(d_qsum, nWorker, d_qsum_total);
		reduceSumArrayKernel<<<16, 128>>>(d_qcount, nWorker, d_qcount_total);
		updateEqKernel<<<1, 1>>>(d_Eq, d_qsum_total, d_qcount_total, nsq);

        // Print progress
		if (iter % MAX(1,maxIter/1000)==0 || iter==maxIter-1)
        {
			fprintf(stderr, "%cOptimizingp progress: %.1lf%%", 13, 
							(float)iter / maxIter * 100);
            fflush(stderr);
        }
	}

	// Get the result
	CUDA_CHECK_RETURN(
			cudaMemcpy(Y.data(), d_Y, sizeof(float)*nn*DIM, cudaMemcpyDeviceToHost));

	// Free memory on GPU
	freeDataInDevice(d_Y, d_I, d_J,
					 d_qsum, d_qcount, 
					 d_qsum_total, d_qcount_total,
					 d_nnStates1, d_nnStates2, d_neStates,
					 d_gsl_de, d_gsl_dn,
					 d_gsl_de_F, d_gsl_dn_F,
					 d_gsl_de_A, d_gsl_dn_A);

    return Y;
}
