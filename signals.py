import scipy.signal as sig
import numpy as np


def apply_fade(signal, fade_length):
    if fade_length == 0:
        return signal

    window_length = 2 * fade_length

    if window_length > len(signal):
        raise RuntimeError("signal must be longer than twice the fade_length for the fade to work")

    window = sig.hann(window_length)

    signal[:fade_length] *= window[:fade_length]
    signal[-fade_length:] *= window[fade_length:]

    return signal


def generate_sine(frequency_hz, length_seconds, sampling_rate, fade_length=100,
                  initial_phase_radians=0):
    length_samples = int(length_seconds * sampling_rate)
    sample_indices = np.arange(0, length_samples)
    time_vector = sample_indices / sampling_rate
    signal = np.sin(2 * np.pi * frequency_hz * time_vector - initial_phase_radians)
    signal = apply_fade(signal, fade_length)
    return signal


def generate_noise(length_samples):
    return np.random.default_rng().uniform(-1, 1, length_samples)
