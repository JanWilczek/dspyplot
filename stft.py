import librosa
import numpy as np


def plot_spectrogram(signal, fs, name):
    stft = librosa.stft(signal)
    spectrogram = np.abs(stft)
    spectrogram_db = librosa.amplitude_to_db(spectrogram, ref=np.max)
    
    plt.figure(figsize=(10,4))
    img = librosa.display.specshow(spectrogram_db, y_axis='log', x_axis='time', sr=fs, cmap='inferno')
    plt.xlabel('Time [s]')
    plt.ylabel('Frequency [Hz]')
    plt.yticks([63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000], ['63', '125', '250', '500', '1k', '2k', '4k', '8k', '16k'])
    plt.colorbar(img, format="%+2.f dBFS")
    output_path = img_output_dir / f'{name}.png'
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    subprocess.run(['cwebp', '-q', '65', '-resize', '800', '0', output_path, '-o', output_path.with_suffix('.webp')])