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
    time = time_vector(length_seconds, sampling_rate)
    signal = np.sin(2 * np.pi * frequency_hz * time - initial_phase_radians)
    signal = apply_fade(signal, fade_length)
    return signal


def time_vector(length_seconds, sampling_rate):
    length_samples = int(length_seconds * sampling_rate)
    sample_indices = np.arange(0, length_samples)
    time = sample_indices / sampling_rate
    return time


def generate_signal(frequency_hz, length_seconds, sampling_rate, fade_length, generator):
    time = time_vector(length_seconds, sampling_rate)
    signal = generator(frequency_hz, time)
    signal = apply_fade(signal, fade_length)
    return signal


def generate_triangle(frequency_hz, length_seconds, sampling_rate, fade_length=100):
    return generate_signal(frequency_hz, length_seconds, sampling_rate, fade_length, triangle)


def generate_sawtooth(frequency_hz, length_seconds, sampling_rate, fade_length=100):
    return generate_signal(frequency_hz, length_seconds, sampling_rate, fade_length, sawtooth)


def triangle(frequency_hz, time):
    return 4 * np.abs(frequency_hz * time - np.floor(frequency_hz * time + 0.5)) - 1


def sawtooth(frequency_hz, time):
    period = 1 / frequency_hz
    return 2 * (time % period) * frequency_hz - 1


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
    sine_amplitude = 0.25
    signal = sine_amplitude * np.sin(f1 * angular_time) + sine_amplitude * np.sin(f2 * angular_time)
    impulse_amplitude = 0.75
    signal[n1] = impulse_amplitude
    signal[n2] = impulse_amplitude
    return time, signal
