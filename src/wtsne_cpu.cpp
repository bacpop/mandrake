// 2021 John Lees, Gerry Tonkin-Hill, Zhirong Yang
// See LICENSE files

#include "wtsne.hpp"

std::shared_ptr<sce_results>
wtsne(const std::vector<uint64_t> &I, const std::vector<uint64_t> &J,
      std::vector<double> &dists, std::vector<double> &weights,
      const double perplexity, const uint64_t maxIter, const uint64_t nRepuSamp,
      const double eta0, const bool bInit, const bool animated,
      const int n_workers, const int n_threads, const unsigned int seed) {
  // Check input
  std::vector<double> Y, P;
  std::tie(Y, P) =
      wtsne_init<double>(I, J, dists, weights, perplexity, n_threads, seed);
  uint64_t nn = weights.size();
  uint64_t ne = P.size();

  // Setup output
  auto results = std::make_shared<sce_results>(animated, n_workers, maxIter);

  // Set up random number generation
  discrete_table<double> node_table(weights, n_threads);
  discrete_table<double> edge_table(P, n_threads);
  pRNG<double> rng_state(
      n_workers, xoshiro_initial_seed<double>(static_cast<uint32_t>(seed)));

  // SNE algorithm
  const int write_per_worker = n_workers * (nRepuSamp + 1);
  const double nsq = nn * (nn - 1);
  double Eq = 1.0;
  unsigned long long int n_clashes = 0;
  results->add_frame(0, Eq, Y); // starting positions

  using namespace std::literals;
  const auto start = std::chrono::steady_clock::now();
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
        double gain[DIM];
        for (int d = 0; d < DIM; d++) {
          gain[d] = eta * g * dY[d];
          double Yk_read_end, Yl_read_end;
#pragma omp atomic capture
          Yk_read_end = Y[d + lk] += gain[d];
#pragma omp atomic capture
          Yl_read_end = Y[d + ll] -= gain[d];
          if (Yk_read_end != Yk_read[d] + gain[d] ||
              Yl_read_end != Yl_read[d] - gain[d]) {
            overwrite = true;
          }
        }
        if (!overwrite) {
          qsum += q;
          qcount++;
        } else {
          // Find another neighbour
          for (int d = 0; d < DIM; d++) {
#pragma omp atomic update
            Y[d + lk] = Y[d + lk] - gain[d];
#pragma omp atomic update
            Y[d + ll] = Y[d + ll] + gain[d];
          }
#pragma omp atomic update
          n_clashes++;
          r--;
        }
      }
    }
    Eq = (Eq * nsq + qsum) / (nsq + qcount);
    results->add_frame(iter, Eq, Y);
    if (iter % MAX(1, maxIter / 1000) == 0) {
      check_interrupts();
      update_progress(iter, maxIter, eta, Eq, write_per_worker, n_clashes);
    }
  }
  const auto end = std::chrono::steady_clock::now();

  results->add_result(maxIter, Eq, Y);
  std::cerr << std::endl
            << "Optimizing done in " << (end - start) / 1s << "s" << std::endl;

  return results;
}
