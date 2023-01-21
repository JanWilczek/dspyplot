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


def generate_noise(length_samples, fade_length=0):
    return apply_fade(np.random.default_rng(seed=1).uniform(-1, 1, length_samples), fade_length)


def zero_pad(signal, zeros_count):
    """Assumes that signal is 1D"""
    return np.concatenate((signal, np.zeros((zeros_count,))))


def generate_two_sines_two_pulses(sampling_rate):
    length_seconds = 1
    time = np.arange(0, length_seconds, 1/sampling_rate)
    f1 = 400
    f2 = 450
    t1 = 0.45
    t2 = 0.5
    n1 = int(t1 * sampling_rate)
    n2 = int(t2 * sampling_rate)
    angular_time = 2 * np.pi * time
    signal = np.sin(f1 * angular_time) + np.sin(f2 * angular_time)
    signal[n1] = 1
    signal[n2] = 1
    return signal
