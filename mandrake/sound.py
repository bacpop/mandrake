# vim: set fileencoding=<utf-8> :
# Copyright 2021 John Lees

# Adds sound to the visualisation

import sys

from tempfile import NamedTemporaryFile
import numpy as np
from scipy.io.wavfile import write
from tqdm import tqdm

from .utils import norm_and_centre

def wave(t):
    x = t % 1.0
    if (x <= 0.25):
        amp = 4.0 * x
    elif (x <= 0.75):
        amp = 2.0 - 4.0 * x
    else:
        amp = 4.0 * x - 4.0
    return amp

def envelope(t, duration):
    x = float(t) / duration
    if (x > 1.0):
        x = 1.0

    attack = 0.025
    decay = 0.1
    sustain = 0.9
    release = 0.3

    if (x < attack):
        amp = 1.0 / attack * x
    elif (x < attack + decay):
        amp = 1.0 - (x - attack) / decay * (1.0 - sustain)
    elif (x < 1.0 - release):
        amp = sustain
    else:
        amp = sustain / release * (1.0 - x)

    return amp

class Oscillator:
    def __init__(self, freq, start, duration):
        self.freq = freq
        self.start = start
        self.duration = duration

    def get_amp(self, t, sample_rate):
        amp = 0
        if t >= self.start:
            amp = wave((t - self.start) / sample_rate * self.freq)
            amp *= envelope(t - self.start, self.duration)
        return amp

def write_wav(results, total_duration, sample_rate=44100):
    # Extract oscillator frequencies from data
    em_prev = np.array(results.get_embedding_frame(0)).reshape(-1, 2)
    norm_and_centre(em_prev)
    freqs = np.zeros((results.n_frames() - 1, 2))
    for frame in range(1, results.n_frames()):
        em_next = np.array(results.get_embedding_frame(frame)).reshape(-1, 2)
        norm_and_centre(em_next)
        freqs[frame - 1, :] = np.max(np.abs(em_prev - em_next), axis=0)
        em_prev = em_next
    # Normalise to 120-1200Hz
    freqs -= np.min(freqs)
    freqs /= np.max(freqs)
    freqs = 120 + 1200 * np.square(freqs)
    print(freqs)

    # Create a list of oscillators across the time series
    oscillators_x = []
    oscillators_y = []
    for idx, freq in enumerate(freqs):
        oscillators_x.append(Oscillator(freq[0], float(idx) / total_duration, sample_rate / 8))
        oscillators_y.append(Oscillator(freq[1], float(idx) / total_duration, sample_rate / 8))

    # Sample from the oscillators across the time series, at the sample rate
    t_series = np.linspace(0., total_duration, int(sample_rate * total_duration))
    amp_series = np.zeros((t_series.shape[0], 2), dtype=np.int16)
    sys.stderr.write("Creating audio")
    for idx, t_point in tqdm(enumerate(t_series), total=t_series.shape[0], unit="samples"):
        for osc_x, osc_y in zip(oscillators_x, oscillators_y):
            amp_series[idx, 0] += osc_x.get_amp(t_point, sample_rate)
            amp_series[idx, 1] += osc_y.get_amp(t_point, sample_rate)

    outfile = NamedTemporaryFile(suffix = ".wav")
    write(outfile.name, sample_rate, amp_series)
    return outfile

