#pragma once

#include <cmath>
#include <vector>

class oscillator {
public:
  oscillator(double freq, double start, double duration)
      : freq_(freq), start_(start), duration_(duration) {}

  double get_amp(double t, double sample_rate) const {
    double amp = 0.0;
    if (t > start_) {
      amp = wave(t, sample_rate);
      amp *= envelope(t);
    }
    return amp;
  }

private:
  double wave(double t, double sample_rate) const {
    double t0 = (t - start_) * freq_ / sample_rate;
    double x = fmod(t0, 1.0);
    double amp = 0.0;
    if (x <= 0.25) {
      amp = 4.0 * x;
    } else if (x <= 0.75) {
      amp = 2.0 - 4.0 * x;
    } else {
      amp = 4.0 * x - 4.0;
    }
    return amp;
  }

  double envelope(double t) const {
    double t0 = t - start_;

    double x = float(t0) / duration_;
    if (x > 1.0) {
      x = 1.0;
    }

    double attack = 0.025;
    double decay = 0.1;
    double sustain = 0.9;
    double release = 0.3;

    double amp = 1.0;
    if (x < attack) {
      amp = 1.0 / attack * x;
    } else if (x < attack + decay) {
      amp = 1.0 - (x - attack) / decay * (1.0 - sustain);
    } else if (x < 1.0 - release) {
      amp = sustain;
    } else {
      amp = sustain / release * (1.0 - x);
    }

    return amp;
  }

  double freq_;
  double start_;
  double duration_;
};

std::vector<double> sample_wave(const std::vector<double> &f_series,
                                const double total_duration,
                                const double sample_rate,
                                const int n_threads) {
  size_t n_steps = static_cast<size_t>(total_duration * sample_rate);
  std::vector<double> amp_wave(n_steps, 0.0);
  std::vector<oscillator> osc_vec;
  for (size_t idx = 0; idx < f_series.size(); ++idx) {
    osc_vec.push_back(oscillator(f_series[idx], idx / total_duration, 1.0 / 8.0));
  }

#pragma omp parallel for schedule(static) num_threads(n_threads)
  for (size_t t = 0; t < n_steps; ++t) {
    for (auto osc = osc_vec.cbegin(); osc != osc_vec.cend(); ++osc) {
      amp_wave[t] += osc->get_amp(t / total_duration, sample_rate);
    }
  }
  return amp_wave;
}
