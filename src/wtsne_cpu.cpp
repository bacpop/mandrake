/*
 ============================================================================
 Author      : Zhirong Yang
 Copyright   : Copyright by Zhirong Yang. All rights are reserved.
 Description : Stochastic Optimization for t-SNE
 ============================================================================
 */

// Modified by John Lees

#include "wtsne.hpp"

std::vector<double>
wtsne(const std::vector<uint64_t> &I, const std::vector<uint64_t> &J,
      std::vector<double> &dists, std::vector<double> &weights,
      const double perplexity, const uint64_t maxIter, const uint64_t nRepuSamp,
      const double eta0, const bool bInit, const int n_workers,
      const int n_threads, const unsigned int seed) {
  // Check input
  std::vector<double> Y, P;
  std::tie(Y, P) =
      wtsne_init<double>(I, J, dists, weights, perplexity, n_threads, seed);
  uint64_t nn = weights.size();
  uint64_t ne = P.size();

  // Set up random number generation
  discrete_table<double> node_table(weights, n_threads);
  discrete_table<double> edge_table(P, n_threads);
  pRNG<double> rng_state(n_workers, std::vector<uint32_t>(1, seed));

  // SNE algorithm
  const double nsq = nn * (nn - 1);
  double Eq = 1.0;
  for (uint64_t iter = 0; iter < maxIter; iter++) {
    double eta = eta0 * (1 - (double)iter / maxIter);
    eta = MAX(eta, eta0 * 1e-4);
    double c = 1.0 / (Eq * nsq);

    double qsum = 0;
    uint64_t qcount = 0;

    double attrCoef = (bInit && iter < maxIter / 10) ? 8 : 2;
    double repuCoef = 2 * c / nRepuSamp * nsq;
#pragma omp parallel for reduction(+ : qsum, qcount) num_threads(n_threads)
    for (int worker = 0; worker < n_workers; worker++) {
      std::vector<double> dY(DIM);
      std::vector<double> Yk_read(DIM);
      std::vector<double> Yl_read(DIM);

      rng_state_t<double> &worker_rng = rng_state.state(worker);
      uint64_t e = edge_table.discrete_draw(worker_rng) % ne;
      uint64_t i = I[e];
      uint64_t j = J[e];

      for (uint64_t r = 0; r < nRepuSamp + 1; r++) {
        uint64_t k, l;
        if (r == 0) {
          k = i;
          l = j;
        } else {
          k = node_table.discrete_draw(worker_rng) % nn;
          l = node_table.discrete_draw(worker_rng) % nn;
        }
        if (k == l) {
          continue;
        }

        uint64_t lk = k * DIM;
        uint64_t ll = l * DIM;
        double dist2 = 0.0;
        for (int d = 0; d < DIM; d++) {
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
        for (int d = 0; d < DIM; d++) {
          double gain = eta * g * dY[d];
          double Yk_read_end, Yl_read_end;
#pragma omp atomic capture
          Yk_read_end = Y[d + lk] += gain;
#pragma omp atomic capture
          Yl_read_end = Y[d + ll] -= gain;
          if (Yk_read_end != Yk_read[d] + gain ||
              Yl_read_end != Yl_read[d] - gain) {
            overwrite = true;
            break;
          }
        }
        if (!overwrite) {
          qsum += q;
          qcount++;
        } else {
          // Find another neighbour
          for (int d = 0; d < DIM; d++) {
#pragma omp atomic write
            Y[d + lk] = Yk_read[d];
#pragma omp atomic write
            Y[d + ll] = Yl_read[d];
          }
          r--;
        }
      }
    }
    Eq = (Eq * nsq + qsum) / (nsq + qcount);
    update_progress(iter, maxIter, eta, Eq);
  }
  std::cerr << std::endl << "Optimizing done" << std::endl;

  return Y;
}
