import numpy as np
import librosa


def dft(signal):
    return np.fft.rfft(signal)


def magnitude_spectrum(signal):
    return np.abs(dft(signal))


def normalized_magnitude_spectrum(signal, db=False):
    spectrum = magnitude_spectrum(signal)
    normalized = spectrum / np.amax(spectrum)

    if db:
        return librosa.amplitude_to_db(normalized)

    return normalized


def two_sided_magnitude_spectrum(signal):
    return np.abs(np.fft.fft(signal))


def magnitude_spectrum_db(signal):
    return librosa.amplitude_to_db(magnitude_spectrum(signal))


def dft_frequencies(dft_length, sampling_rate):
    return np.fft.rfftfreq(dft_length, 1 / sampling_rate)
