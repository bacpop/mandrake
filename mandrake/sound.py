# vim: set fileencoding=<utf-8> :
# Copyright 2021 John Lees

# Adds sound to the visualisation

import subprocess
from tempfile import NamedTemporaryFile
import numpy as np
from scipy.io.wavfile import write

from .utils import norm_and_centre

from SCE import gen_audio

def write_wav(results, video_file, total_duration, sample_rate=44100, threads=1):
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
    x_audio = np.array(gen_audio(list(freqs[:, 0]), total_duration, sample_rate, threads), dtype=np.int16)
    y_audio = np.array(gen_audio(list(freqs[:, 1]), total_duration, sample_rate, threads), dtype=np.int16)

    #outfile = NamedTemporaryFile(suffix=".wav", delete=False)
    #write(outfile.name, sample_rate, np.column_stack((x_audio, y_audio)))
    outfile = "tmp.wav"
    write(outfile, sample_rate, np.column_stack((x_audio, y_audio)))
    subprocess.run(["ffmpeg", "-i", video_file, "-i", outfile, "-c:v",
     "copy", "-map", "0:v:0", "-map", "1:a:0", "-c:a aac", "-b:a 192k",
      "tmp.mp4"], check=True)

