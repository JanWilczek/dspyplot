from pathlib import Path
import matplotlib.pyplot as plt
import style


def prepare_output_path(path, stem_suffix: str):
    return path.with_name(path.stem + stem_suffix + style.img_file_suffix)


def save(output_path, suffix):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(prepare_output_path(output_path, suffix), **style.save_params)


def save_spectrum(output_path):
    save(output_path, '_spectrum')


def save_signal(output_path):
    save(output_path, '_signal')


def _stem(points):
    markerline, stemlines, baseline = plt.stem(points, **style.stem_params)
    plt.setp(markerline, 'color', style.color)
    plt.setp(stemlines, 'color', style.color)
    plt.setp(baseline, visible=False)


def stem_signal_and_save(signal, output_path: Path):
    samples_count = signal.shape[0]

    plt.figure(figsize=(12,6))
    _stem(signal)
    plt.yticks([-1, 0, 1])
    plt.xticks([])
    plt.xlim([0, samples_count])
    plt.xlabel('sample index $n$')
    plt.ylabel('amplitude')
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    save_signal(output_path)


def stem_spectrum_and_save(magnitude_spectrum, output_path: Path):
    plt.figure(figsize=(12, 6))
    _stem(magnitude_spectrum)
    plt.yticks([])
    plt.xlim([0, magnitude_spectrum.shape[0]])
    plt.hlines(0, 0, magnitude_spectrum.shape[0], colors='k')
    plt.xlabel('frequency bin index $k$')
    plt.ylabel('magnitude')
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    save_spectrum(output_path)
    