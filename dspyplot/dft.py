import numpy as np
from .signals import amplitude2db


def dft(signal):
    """
    Compute the unnormalized right-side DFT of the input (only the positive frequencies).

    From the docs: "n input points produce n/2+1 complex output points."
    """
    return np.fft.rfft(signal)


def idft(spectrum):
    """
    Compute the inverse right-side DFT normalized by the length of the DFT.

    From the docs: "for an output of n points uses n/2+1 input points(...)
    If n is not given, it is taken to be 2*(m-1) where m is the length of the input
    along the axis specified by axis."
    """
    return np.fft.irfft(spectrum)


def magnitude_spectrum(signal):
    return np.abs(dft(signal))


def normalized_magnitude_spectrum(signal, db=False):
    spectrum = magnitude_spectrum(signal)
    normalized = spectrum / np.amax(spectrum)

    if db:
        return amplitude2db(normalized)

    return normalized


def two_sided_magnitude_spectrum(signal):
    return np.abs(np.fft.fft(signal))


def magnitude_spectrum_db(signal):
    return amplitude2db(magnitude_spectrum(signal))


def dft_frequencies(spectrum_length, sampling_rate):
    return np.fft.rfftfreq(2 * spectrum_length - 1, 1 / sampling_rate)


def two_sided_dft_frequencies(spectrum_length, sampling_rate):
    return 2 * np.fft.rfftfreq(2 * spectrum_length - 1, 1 / sampling_rate)
