/*
 ============================================================================
 Author      : Zhirong Yang
 Copyright   : Copyright by Zhirong Yang. All rights are reserved.
 Description : Stochastic Optimization for t-SNE
 ============================================================================
 */

// Modified by John Lees

#include <stdio.h>
#include <string.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <vector>
#include <iostream>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

#ifndef DIM
#define DIM 2
#endif

#define MAX(a,b) ( (a) > (b) ? (a) : (b) )
#define MIN(a,b) ( (a) < (b) ? (a) : (b) )

std::vector<double> wtsne(std::vector<long long>& I,
           std::vector<long long>& J,
           std::vector<double>& P,
           std::vector<double>& weights,
           long long maxIter, 
           long long workerCount, 
           long long nRepuSamp,
           double eta0,
           bool bInit)
{
    // Check input
    if (I.size() != J.size() || I.size() != P.size() || J.size() != P.size())
    {
        std::cerr << "Mismatching sizes in input vectors" << std::endl;
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
    std::vector<double> Y(nn*DIM);
    for (long long i = 0; i < nn; i++)
        for (long long d = 0; d < DIM; d++)
            Y[d + i*DIM] = rand() * 1e-4 / RAND_MAX;

    // Set up random number generation
    const gsl_rng_type * gsl_T;
    gsl_rng_env_setup();
    gsl_T = gsl_rng_default;
    gsl_rng * gsl_r_nn = gsl_rng_alloc(gsl_T);
    gsl_rng * gsl_r_ne = gsl_rng_alloc(gsl_T);
    gsl_rng_set(gsl_r_nn, 314159);
    gsl_rng_set(gsl_r_ne, 271828);

    gsl_ran_discrete_t * gsl_de = gsl_ran_discrete_preproc(ne, P.data());
    gsl_ran_discrete_t * gsl_dn = gsl_ran_discrete_preproc(nn, weights.data());

    // SNE algorithm
    const double nsq = nn * (nn-1);
    double Eq = 1.0;
    for (long long iter=0; iter<maxIter; iter++)
    {
        double eta = eta0 * (1-(double)iter/maxIter);
        eta = MAX(eta, eta0 * 1e-4);
        double c = 1.0 / (Eq * nsq);

        double qsum = 0;
        long long qcount = 0;

        double attrCoef = (bInit && iter<maxIter/10) ? 8 : 2;
        double repuCoef = 2 * c / nRepuSamp * nsq;
        #pragma omp parallel for reduction(+:qsum,qcount)
        for (long long worker=0; worker<workerCount; worker++)
        {
            std::vector<double> dY(DIM);

            long long e = gsl_ran_discrete(gsl_r_ne, gsl_de) % ne;
            long long i = I[e];
            long long j = J[e];

            for (long long r=0; r<nRepuSamp+1; r++)
            {
                long long k, l;
                if (r==0)
                {
                    k = i;
                    l = j;
                }
                else
                {
                    k = gsl_ran_discrete(gsl_r_nn, gsl_dn) % nn;
                    l = gsl_ran_discrete(gsl_r_nn, gsl_dn) % nn;
                }
                if (k == l) continue;

                long long lk = k * DIM;
                long long ll = l * DIM;
                double dist2 = 0.0;
                for (long long d=0; d<DIM; d++)
                {
                    dY[d] = Y[d+lk]-Y[d+ll];
                    dist2 += dY[d] * dY[d];
                }
                double q = 1.0 / (1 + dist2);

                double g;
                if (r==0)
                    g = -attrCoef * q;
                else
                    g = repuCoef * q * q;

                for (long long d=0; d<DIM; d++)
                {
                    double gain = eta * g * dY[d];
                    Y[d+lk] += gain;
                    Y[d+ll] -= gain;
                }
                qsum += q;
                qcount++;
            }
        }
        Eq = (Eq * nsq + qsum) / (nsq + qcount);

        if (iter % MAX(1,maxIter/1000)==0)
        {
            fprintf(stderr, "%cOptimizing\t eta=%f Progress: %.1lf%%, Eq=%.20f", 13, eta, (double)iter+1 / maxIter * 100, 1.0/(c*nsq));
            fflush(stderr);
        }
    }
    std::cerr << std::endl << "Optimizing done" << std::endl;

    // Free memory from GSL functions
    gsl_ran_discrete_free(gsl_de);
    gsl_ran_discrete_free(gsl_dn);
    gsl_rng_free(gsl_r_nn);
    gsl_rng_free(gsl_r_ne);

    return(Y);
}

PYBIND11_MODULE(SCE, m)
{
  m.doc() = "Stochastic cluster embedding";

  // Exported functions
  m.def("wtsne", &wtsne, py::return_value_policy::take_ownership, "Run stochastic cluster embedding", 
        py::arg("I_vec"),
        py::arg("J_vec"),
        py::arg("P_vec"),
        py::arg("weights"),
        py::arg("maxIter"),
        py::arg("workerCount") = 1,
        py::arg("nRepuSamp") = 5,
        py::arg("eta0") = 1,
        py::arg("bInit") = 0);
}

