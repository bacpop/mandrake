#pragma once

#include <stack>
#include <stdexcept>

#include "rng.hpp"
#include "vector_norm.hpp"

#ifdef __NVCC__
template <typename real_t> struct discrete_table_ptrs {
  uint64_t K;
  real_t *F;
  uint64_t *A;
};

template <typename real_t> struct discrete_table_device {
  device_array<real_t> F;
  device_array<uint64_t> A;
};
#endif

template <typename real_t, typename real_t_p = real_t> class discrete_table {
public:
  discrete_table(std::vector<real_t_p> probs, const int n_threads = 1)
      : K(probs.size()), F(K), A(K) {
    if (probs.size() < 1) {
      throw std::runtime_error("Probability table has too few values");
    }
    normalise_vector(probs, true, n_threads);

    /* This code is based on randist/discrete.c from the GSL library
     *
     * Implements an O(N) version of Walker's algorithm to set lookup
     * tables A and F. These are then O(1) to draw from
     *
     * Based on: Alastair J Walker, An efficient method for generating
     * discrete random variables with general distributions, ACM Trans
     * Math Soft 3, 253-256 (1977).  See also: D. E. Knuth, The Art of
     * Computer Programming, Volume 2 (Seminumerical algorithms), 3rd
     * edition, Addison-Wesley (1997), p120.
     */

    /* Now create the Bigs and the Smalls */
    std::stack<real_t> Bigs, Smalls;
    real_t_p mean = static_cast<real_t>(1.0) / K;

    /* Temporarily use A[k] to indicate small or large */
    for (uint64_t k = 0; k < K; ++k) {
      if (probs[k] < mean) {
        A[k] = 0;
      } else {
        A[k] = 1;
      }
    }

    for (uint64_t k = 0; k < K; ++k) {
      if (A[k]) {
        Bigs.push(static_cast<real_t>(k));
      } else {
        Smalls.push(static_cast<real_t>(k));
      }
    }

    /* Now work through the smalls */
    while (Smalls.size() > 0) {
      real_t s = Smalls.top();
      Smalls.pop();
      if (Bigs.size() == 0) {
        A[s] = s;
        F[s] = static_cast<real_t>(1.0);
        continue;
      }
      if (Bigs.size() == 0) {
        throw std::runtime_error("Bigs stack has been emptied");
      }
      real_t b = Bigs.top();
      Bigs.pop();
      A[s] = b;
      F[s] = K * probs[s];

      real_t_p d = mean - probs[s];
      probs[s] += d; /* now E[s] == mean */
      probs[b] -= d;
      if (probs[b] < mean) {
        Smalls.push(b); /* no longer big, join ranks of the small */
      } else if (probs[b] > mean) {
        Bigs.push(b); /* still big, put it back where you found it */
      } else {
        /* E[b]==mean implies it is finished too */
        A[b] = b;
        F[b] = static_cast<real_t>(1.0);
      }
    }
    while (Bigs.size() > 0) {
      real_t b = Bigs.top();
      Bigs.pop();
      A[b] = b;
      F[b] = static_cast<real_t>(1.0);
    }
    /* Stacks have been emptied, and A and F have been filled */
    if (Smalls.size() != 0) {
      throw std::runtime_error("Smalls stack has not been emptied");
    }
#pragma omp parallel for schedule(static) num_threads(n_threads)
    /* This is the Knuth convention */
    for (uint64_t k = 0; k < K; ++k) {
      F[k] += k;
      F[k] /= K;
    }
  }

  uint64_t size() const { return K; }
  std::vector<real_t> &F_table() { return F; }
  std::vector<uint64_t> &A_table() { return A; }

  uint64_t discrete_draw(rng_state_t<real_t> &rng_state) {
    real_t u = unif_rand<real_t>(rng_state);
    uint64_t c = u * K;
    real_t f = F[c];

    real_t draw;
    if (f == static_cast<real_t>(1.0) || u < f) {
      draw = c;
    } else {
      draw = A[c];
    }
    return draw;
  }

private:
  uint64_t K;
  std::vector<real_t> F;
  std::vector<uint64_t> A;
};

#ifdef __NVCC__
template <typename real_t>
DEVICE uint64_t discrete_draw(rng_state_t<real_t> &rng_state,
                              const discrete_table_ptrs<real_t> &unif_table) {
  real_t u = unif_rand<real_t>(rng_state);
  uint64_t c = u * unif_table.K;
  real_t f = unif_table.F[c];

  real_t draw;
  if (f == static_cast<real_t>(1.0) || u < f) {
    draw = c;
  } else {
    draw = unif_table.A[c];
  }
  __syncwarp();
  return static_cast<uint64_t>(draw);
}
#endif
