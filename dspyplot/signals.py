import scipy.signal as sig
import numpy as np

OCTAVE_BANDS = np.asarray([63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000])
OCTAVE_BANDS_LABELS = [str(f) for f in OCTAVE_BANDS]


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


def delay_signal(signal, delay):
    delayed_signal = np.concatenate((np.zeros((delay,)), signal))
    return delayed_signal


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


def generate_signal(frequency_hz, length_seconds, sampling_rate, fade_length, generator, **kwargs):
    time = time_vector(length_seconds, sampling_rate)
    signal = generator(frequency_hz, time, **kwargs)
    signal = apply_fade(signal, fade_length)
    return signal


def generate_triangle(frequency_hz, length_seconds, sampling_rate, fade_length=100):
    return generate_signal(frequency_hz, length_seconds, sampling_rate, fade_length, triangle)


def generate_sawtooth(frequency_hz, length_seconds, sampling_rate, fade_length=100):
    return generate_signal(frequency_hz, length_seconds, sampling_rate, fade_length, sawtooth)


def generate_square(frequency_hz, length_seconds, sampling_rate, fade_length=100, harmonics_count=13):
    return generate_signal(frequency_hz, length_seconds, sampling_rate, fade_length, square, harmonics_count=harmonics_count)


def generate_pulse(frequency_hz, length_seconds, sampling_rate, fade_length=100, duty_cycle=0.2, harmonics_count=14):
    return generate_signal(frequency_hz, length_seconds, sampling_rate, fade_length, pulse, duty_cycle=duty_cycle, harmonics_count=harmonics_count)


def triangle(frequency_hz, time):
    return 4 * np.abs(frequency_hz * time - np.floor(frequency_hz * time + 0.5)) - 1


def sawtooth(frequency_hz, time):
    period = 1 / frequency_hz
    return 2 * (time % period) * frequency_hz - 1


def square(frequency_hz, time, harmonics_count=13):
    phase = 2 * np.pi * frequency_hz * time
    harmonics_count = harmonics_count // 2
    waveform = np.zeros_like(phase)
    for k in range(1, harmonics_count + 1):
        waveform += 4 / np.pi * (2 * k - 1) ** -1 * np.sin((2 * k - 1) * phase)
    return waveform


def pulse(frequency_hz, time, duty_cycle=0.2, harmonics_count=14):
    phase = 2 * np.pi * frequency_hz * time
    waveform = (2 * duty_cycle - 1) * np.ones_like(phase)
    for k in range(1, harmonics_count + 1):
        waveform += 4 / (k * np.pi) * np.sin(np.pi * k * duty_cycle) * np.cos(k * phase - np.pi * k * duty_cycle)
    return waveform


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


def generate_farina_sweep(from_digital_frequency, to_digital_frequency, duration_in_samples):
    amplitude = 1.0
    factor = 2 * np.pi * from_digital_frequency * duration_in_samples / np.log(to_digital_frequency / from_digital_frequency)
    exponent = np.log(to_digital_frequency / from_digital_frequency) / duration_in_samples
    farina_sweep = amplitude * np.sin(factor * (np.exp(np.arange(0, duration_in_samples) * exponent - 1)))
    return farina_sweep


def db2amplitude(db):
    return np.power(10, db / 20)


def amplitude2db(amplitude):
    return 20 * np.log10(np.maximum(amplitude, 1e-6))


class UniversalCombFilter:
    def __init__(self, delay: int, blend: float, feedback: float, feedforward: float,
                 name: str = ''):
        self.delay = delay
        self.blend = blend
        self.feedback = feedback
        self.feedforward = feedforward
        self.name = name

    def ba(self):
        b = np.zeros((self.delay + 1,))
        a = np.zeros((self.delay + 1,))
        b[0] = self.blend
        b[self.delay] = self.feedforward
        a[0] = 1
        a[self.delay] = - self.feedback
        return b, a
