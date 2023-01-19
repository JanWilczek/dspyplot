from pathlib import Path
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from plot import save, frequency_ticks


def plot_spectrogram_and_save(signal, fs, output_path: Path):
    stft = librosa.stft(signal)
    spectrogram = np.abs(stft)
    spectrogram_db = librosa.amplitude_to_db(spectrogram, ref=np.max)
    
    plt.figure(figsize=(10,4))
    img = librosa.display.specshow(spectrogram_db, y_axis='log', x_axis='time', sr=fs, cmap='inferno')
    plt.xlabel('Time [s]')
    plt.ylabel('Frequency [Hz]')
    plt.yticks(*frequency_ticks(min_frequency=60, max_frequency=(fs/2)))
    plt.colorbar(img, format="%+2.f dBFS")

    save(output_path, '_spectrogram')
