/*
 ============================================================================
 Author      : Zhirong Yang
 Copyright   : Copyright by Zhirong Yang. All rights are reserved.
 Description : Stochastic Optimization for t-SNE
 ============================================================================
 */

// Modified by John Lees

#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>

#include "wtsne.hpp"

std::vector<double> wtsne(const std::vector<uint64_t> &I,
                          const std::vector<uint64_t> &J,
                          std::vector<double> &dists,
                          std::vector<double> &weights, const double perplexity,
                          const uint64_t maxIter, const uint64_t nRepuSamp,
                          const double eta0, const bool bInit,
                          const int n_threads, const int seed)
{
  // Check input
  std::vector<double> Y, P;
  std::tie(Y, P) =
      wtsne_init<double>(I, J, dists, weights, perplexity, n_threads, seed);
  long long nn = weights.size();
  long long ne = P.size();

  // Set up random number generation
  const gsl_rng_type *gsl_T;
  gsl_rng_env_setup();
  gsl_T = gsl_rng_default;
  gsl_rng *gsl_r_nn = gsl_rng_alloc(gsl_T);
  gsl_rng *gsl_r_ne = gsl_rng_alloc(gsl_T);
  gsl_rng_set(gsl_r_nn, seed);
  gsl_rng_set(gsl_r_ne, seed << 1); // not ideal seeding

  gsl_ran_discrete_t *gsl_de = gsl_ran_discrete_preproc(ne, P.data());
  gsl_ran_discrete_t *gsl_dn = gsl_ran_discrete_preproc(nn, weights.data());

  // SNE algorithm
  const double nsq = nn * (nn - 1);
  double Eq = 1.0;
  for (long long iter = 0; iter < maxIter; iter++)
  {
    double eta = eta0 * (1 - (double)iter / maxIter);
    eta = MAX(eta, eta0 * 1e-4);
    double c = 1.0 / (Eq * nsq);

    double qsum = 0;
    long long qcount = 0;

    double attrCoef = (bInit && iter < maxIter / 10) ? 8 : 2;
    double repuCoef = 2 * c / nRepuSamp * nsq;
#pragma omp parallel for reduction(+ \
                                   : qsum, qcount) num_threads(n_threads)
    for (long long worker = 0; worker < n_threads; worker++)
    {
      std::vector<double> dY(DIM);
      std::vector<double> Yk_read(DIM);
      std::vector<double> Yl_read(DIM);

      long long e = gsl_ran_discrete(gsl_r_ne, gsl_de) % ne;
      long long i = I[e];
      long long j = J[e];

      for (long long r = 0; r < nRepuSamp + 1; r++)
      {
        // fprintf(stderr, "r: %d", r);
        // fflush(stderr);
        long long k, l;
        if (r == 0)
        {
          k = i;
          l = j;
        }
        else
        {
          k = gsl_ran_discrete(gsl_r_nn, gsl_dn) % nn;
          l = gsl_ran_discrete(gsl_r_nn, gsl_dn) % nn;
        }
        if (k == l)
          continue;

        long long lk = k * DIM;
        long long ll = l * DIM;
        double dist2 = 0.0;
        for (long long d = 0; d < DIM; d++)
        {
#pragma omp atomic read
          Yk_read[d] = Y[d + lk];
#pragma omp atomic read
          Yl_read[d] = Y[d + ll];
          dY[d] = Yk_read[d] - Yl_read[d];
          dist2 += dY[d] * dY[d];
        }
        double q = 1.0 / (1 + dist2);

        double g;
        if (r == 0)
          g = -attrCoef * q;
        else
          g = repuCoef * q * q;

        bool overwrite = false;
        for (long long d = 0; d < DIM; d++)
        {
          double gain = eta * g * dY[d];
          double Yk_read_end, Yl_read_end;

          Yk_read_end = Y[d + lk];
          Y[d + lk] =
              Yk_read_end == Yk_read[d] ? Yk_read[d] + gain : Yk_read[d];

          Yl_read_end = Y[d + ll];
          Y[d + ll] =
              Yl_read_end == Yl_read[d] ? Yl_read[d] - gain : Yl_read[d];

          if (Yl_read_end != Yl_read[d] || Yk_read_end != Yk_read[d])
          {
            overwrite = true;
          }
        }
        if (!overwrite)
        {
          qsum += q;
          qcount++;
        }
        else
        {
          for (int d = 0; d < DIM; d++)
          {
#pragma atomic write
            Y[d + lk] = Yk_read[d];
#pragma atomic write
            Y[d + ll] = Yl_read[d];
          }
        }
      }
    }
    Eq = (Eq * nsq + qsum) / (nsq + qcount);

    if (iter % MAX(1, maxIter / 1000) == 0 || iter == maxIter - 1)
    {
      fprintf(stderr, "%cOptimizing (CPU)\t eta=%f Progress: %.1lf%%, Eq=%.20f",
              13, eta, (double)iter / maxIter * 100, 1.0 / (c * nsq));
      fflush(stderr);
    }
  }
  std::cerr << std::endl
            << "Optimizing done" << std::endl;

  // Free memory from GSL functions
  gsl_ran_discrete_free(gsl_de);
  gsl_ran_discrete_free(gsl_dn);
  gsl_rng_free(gsl_r_nn);
  gsl_rng_free(gsl_r_ne);

  return (Y);
}
