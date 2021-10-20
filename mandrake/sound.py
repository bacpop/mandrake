# vim: set fileencoding=<utf-8> :
# Copyright 2021 John Lees

# Adds sound to the visualisation

import os
import subprocess
from tempfile import mkstemp
import numpy as np
from scipy.io.wavfile import write as write_wav

from .utils import norm_and_centre

from SCE import gen_audio

def encode_audio(results, video_file, total_duration, sample_rate=44100, threads=1):
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
    freqs = 120 + 1200 * freqs
    # Encode
    x_audio = _freq_to_wave(list(freqs[:, 0]), total_duration, sample_rate, threads)
    y_audio = _freq_to_wave(list(freqs[:, 1]), total_duration, sample_rate, threads)
    audio = np.column_stack((x_audio, y_audio))

    x_audio = np.array(gen_audio(list(freqs[:, 0]), total_duration, sample_rate, threads))
    x_audio *= np.iinfo(np.int16).max / np.max(np.abs(x_audio))
    x_audio = x_audio.astype(np.int16, copy=False)
    y_audio = np.array(gen_audio(list(freqs[:, 1]), total_duration, sample_rate, threads))
    y_audio *= np.iinfo(np.int16).max / np.max(np.abs(y_audio))
    y_audio = y_audio.astype(np.int16, copy=False)

    # Save the audio as an uncompressed WAV
    wav_tmp = mkstemp(suffix=".wav")[1]
    write_wav(wav_tmp, sample_rate, audio)
    # Compress (aac) and add to the video
    vid_tmp = mkstemp(suffix=".mp4")[1]
    try:
        subprocess.run("ffmpeg -y -i " + video_file + " -i " + wav_tmp + \
          " -c:v copy -map 0:v:0 -map 1:a:0 -c:a aac -b:a 192k " + \
          vid_tmp, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        os.remove(vid_tmp)
        os.remove(wav_tmp)
        raise e

    # Sort out tmp files so output is correct
    os.rename(vid_tmp, video_file)
    os.remove(wav_tmp)

# internal functions

# Create a list of oscillators across the time series
# Normalise amplitude based on 16-bit signed ints
def _freq_to_wave(freq_list, duration, sample_rate, threads):
    audio = np.array(gen_audio(freq_list, duration, sample_rate, threads))
    audio *= np.iinfo(np.int16).max / np.max(np.abs(audio))
    audio = audio.astype(np.int16, copy=False)
    return audio