from pathlib import Path
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from .plot import save, frequency_ticks


def plot_spectrogram_and_save(
    signal, fs, output_path: Path, fft_size=2048, hop_size=None, window_size=None
):

    # default values taken from the librosa documentation
    if not window_size:
        window_size = fft_size

    if not hop_size:
        hop_size = window_size // 4

    stft = librosa.stft(
        signal,
        n_fft=fft_size,
        hop_length=hop_size,
        win_length=window_size,
        center=False,
    )
    spectrogram = np.abs(stft)
    spectrogram_db = librosa.amplitude_to_db(spectrogram, ref=np.max)

    plt.figure(figsize=(10, 4))
    img = librosa.display.specshow(
        spectrogram_db,
        y_axis="log",
        x_axis="time",
        sr=fs,
        hop_length=hop_size,
        cmap="inferno",
    )
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")
    plt.yticks(*frequency_ticks(min_frequency=60, max_frequency=(fs / 2)))
    plt.colorbar(img, format="%+2.f dBFS")

    save(
        output_path,
        f"_spectrogram_WINLEN={window_size}_HOPLEN={hop_size}_NFFT={fft_size}",
    )
