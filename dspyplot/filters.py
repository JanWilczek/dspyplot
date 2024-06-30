import numpy as np


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
