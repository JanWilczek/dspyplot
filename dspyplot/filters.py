import numpy as np
import scipy.signal as sig


def butter(cutoff_frequency_hz, sampling_rate, q, normalized=False):
    """
    Calculates the numerator and denominator coeffcients of the 2nd-order Butterworth lowpass
    based on manual digitization via the bilinear transformation.
    :param cutoff_frequency_hz: cutoff frequency of the filter in hertz
    :param sampling_rate: sampling rate in hertz
    :param q: the Q-factor of the filter
    :param normalized: if True, then filter coefficients will be normalized by a0 so that a0 = 1.
    :return: b, a numerator and denominator coefficients respectively of a digital transfer function
    (see scipy.signal.butter for 'ba' output)
    """
    k = np.tan(np.pi * cutoff_frequency_hz / sampling_rate)
    b0 = k**2
    b1 = 2 * k**2
    b2 = k**2
    a0 = 1 + k / q + k**2
    a1 = 2 * k**2 - 2
    a2 = 1 - k / q + k**2

    b = [b0, b1, b2]
    a = [a0, a1, a2]

    if normalized:
        b /= a0
        a /= a0

    return b, a


def pink_noise_filter(sample_rate):
    start = 20
    frequencies = np.concatenate(
        (
            np.array([0]),
            start * np.power(2, np.arange(0, np.log2(sample_rate / (2 * start)))),
            np.array([sample_rate / 2]),
        ),
    )
    # gain is sqrt(power)
    gain = np.concatenate(
        (np.array([1]), np.sqrt(start) / np.sqrt(frequencies[1:-1]), np.array([0]))
    )
    normalized_frequency = frequencies / (sample_rate / 2)
    h = sig.firwin2(numtaps=2000, freq=normalized_frequency, gain=gain)
    return h
