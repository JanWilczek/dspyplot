import numpy as np
import matplotlib.pyplot as plt

from dspyplot.filters import pink_noise_filter
from .context import dspyplot
from dspyplot import signals
from dspyplot.constants import img_output_path
from dspyplot.dft import magnitude_spectrum, dft_frequencies
from dspyplot.plot import (
    plot_spectrum_and_save,
    plot_magnitude_response_and_save,
    plot_digital_magnitude_responses_in_octaves_and_save,
)

IMG_OUTPUT_PATH = img_output_path("test")


def plot_pink_noise_spectrum():
    sample_rate = 48000
    pink_noise = signals.generate_pink_noise(sample_rate, sample_rate)

    frequencies = np.arange(1, sample_rate / 2, 100)
    spectrum = magnitude_spectrum(pink_noise)
    spectral_envelope = np.sqrt(1 / frequencies) * np.amax(spectrum)
    plot_spectrum_and_save(
        spectrum,
        IMG_OUTPUT_PATH / "pink_noise",
        frequencies=dft_frequencies(spectrum.size, sample_rate),
        extra_command=lambda: plt.plot(frequencies, spectral_envelope),
    )

    plot_digital_magnitude_responses_in_octaves_and_save(
        [pink_noise_filter(sample_rate)],
        [[1]],
        sample_rate,
        IMG_OUTPUT_PATH / "pink_noise",
        db=True,
        ylim=[-30, 3],
        xlim=[20, sample_rate / 2],
    )


def main():
    plot_pink_noise_spectrum()


if __name__ == "__main__":
    main()
