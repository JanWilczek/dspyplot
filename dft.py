import numpy as np
import librosa


def dft(signal):
    return np.fft.rfft(signal)


def magnitude_spectrum(signal):
    return np.abs(dft(signal))


def magnitude_spectrum_db(signal):
    return librosa.amplitude_to_db(magnitude_spectrum(signal))
