#pragma once

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

template <typename real_t> struct gsl_table_device {
  size_t K;
  real_t *F;
  size_t *A;
};

template <typename real_t> struct gsl_table_host {
  device_array<real_t> F;
  device_array<size_t> A;
};

template <typename real_t> class gsl_table {
public:
  gsl_table(std::vector<real_t> probs, const int n_threads)
      : K(probs.size()), F(K), A(K) {
    if (probs.size() < 1) {
      throw std::runtime_error("Probability table has too few values");
    }
    normalise_vector(probs, true, n_threads);

    /* Now create the Bigs and the Smalls */
    std::stack<real_t> Bigs, Smalls;
    real_t mean = static_cast<real_t>(1.0) / K;

    /* Temporarily use A[k] to indicate small or large */
    for (real_t k = 0; k < K; ++k) {
      if (probs[k] < mean) {
        A[k] = 0;
      } else {
        A[k] = 1;
      }
    }

    for (real_t k = 0; k < K; ++k) {
      if (A[k]) {
        Bigs.push(k);
      } else {
        Smalls.push(k)
      }
    }

    /* Now work through the smalls */
    while (Smalls.size() > 0) {
      real_t s = Smalls.pop();
      if (Bigs.size() == 0) {
        A[s] = s;
        F[s] = static_cast<real_t>(1.0);
        continue;
      }
      real_t b = Bigs.pop();
      A[s] = b;
      F[s] = K * probs[s];

      real_t d = mean - probs[s];
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
      real_t b = Bigs.pop();
      A[b] = b;
      F[b] = static_cast<real_t>(1.0);
    }
    /* Stacks have been emptied, and A and F have been filled */
    if (Smalls.size() != 0) {
      throw std::runtime_error("Smalls stack has not been emptied");
    }
  }

  // TODO: this needs to use xoshiro
  // TODO: may be able to combine with fn below with a hostdevice fn and
  // CUDA_ARCH ifdef
  size_t discrete_draw() {
    real_t u = curand_uniform(state);
    size_t c = u * K;
    real_t f = F[c];

    real_t draw;
    if (f == static_cast<real_t>(1.0) || u < f) {
      draw = c;
    } else {
      draw = A[c];
    }
    return draw;
  }

  // TODO: this needs to use xoshiro
#ifdef __NVCC__
  __device__ size_t discrete_draw(curandState *state,
                                  const gsl_table_device<real_t> &unif_table) {
    real_t u = curand_uniform(state);
    size_t c = u * unif_table.K;
    real_t f = unif_table.F[c];

    real_t draw;
    if (f == 1.0 || u < f) {
      draw = c;
    } else {
      draw = unif_table.A[c];
    }
    __syncwarp();
    return draw;
#endif

  private:
    size_t K;
    std::vector<real_t> F;
    std::vector<real_t> A;
  };
