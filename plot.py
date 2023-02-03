from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import style


class PlotPeriodCommand:
    def __init__(self, period_start, period_length, arrows_y=1.2):
        self.vlines_style = dict(linestyle='--', color='grey')
        self.arrow_style = dict(length_includes_head=True, head_width=0.04, head_length=0.03, color='k')
        self.period_start = period_start
        self.period_length = period_length
        self.period_end = self.period_start + self.period_length
        self.arrow_y = arrows_y
        self.label_x = np.mean([self.period_start, self.period_end])
        self.label_y = self.arrow_y * 1.1

    def __call__(self):
        plt.vlines(self.period_start, 0, self.arrow_y, **self.vlines_style)
        plt.vlines(self.period_end, 0, self.arrow_y, **self.vlines_style)
        plt.arrow(self.period_start, self.arrow_y, self.period_length, 0, **self.arrow_style)
        plt.arrow(self.period_end, self.arrow_y, -self.period_length, 0, **self.arrow_style)
        plt.text(self.label_x, self.label_y, '$T$')


def prepare_output_path(path, stem_suffix: str):
    return path.with_name(path.stem + stem_suffix + style.img_file_suffix)


def save(output_path, suffix=''):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(prepare_output_path(output_path, suffix), **style.save_params)


def save_spectrum(output_path):
    save(output_path, '_spectrum')


def save_signal(output_path):
    save(output_path, '_signal')


def frequency_ticks(min_frequency=None, max_frequency=None):
    xticks = np.array([31.25, 62.5, 125, 250, 500, 1000, 2000, 4000, 8000, 16000])
    xtick_labels = np.array(['31.25', '62.5', '125', '250', '500', '1k', '2k', '4k', '8k', '16k'])

    if min_frequency:
        min_filter = xticks >= min_frequency
        xticks = xticks[min_filter]
        xtick_labels = xtick_labels[min_filter]

    if max_frequency:
        max_filter = xticks <= max_frequency
        xticks = xticks[max_filter]
        xtick_labels = xtick_labels[max_filter]

    return xticks, xtick_labels


def _stem(points, bin_indices=None):
    if bin_indices is not None:
        markerline, stemlines, baseline = plt.stem(bin_indices, points, **style.stem_params)
    else:
        markerline, stemlines, baseline = plt.stem(points, **style.stem_params)
    plt.setp(markerline, 'color', style.color)
    plt.setp(stemlines, 'color', style.color)
    plt.setp(baseline, visible=False)


def stem_signal_and_save(signal, output_path: Path, show_xticks=False):
    samples_count = signal.shape[0]

    plt.figure(figsize=(12, 6))
    _stem(signal)
    plt.yticks([-1, 0, 1])
    if not show_xticks:
        plt.xticks([])
    xlim = [-0.5, samples_count - 0.5]
    plt.xlim(xlim)
    plt.hlines(0, xlim[0], xlim[-1], colors='k')
    plt.xlabel('sample index $n$')
    plt.ylabel('amplitude')
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    save_signal(output_path)
    plt.close()


def stem_spectrum_and_save(magnitude_spectrum, output_path: Path, bin_indices=None,
                           yticks=None, ytick_labels=None, ylabel='magnitude',
                           xlabel='frequency bin index $k$', xticks=None):
    plt.figure(figsize=(12, 6))
    _stem(magnitude_spectrum, bin_indices)
    plt.yticks(yticks, ytick_labels)
    plt.xticks(xticks)
    if bin_indices is None:
        plt.xlim([0, magnitude_spectrum.shape[0]])
        plt.hlines(0, 0, magnitude_spectrum.shape[0], colors='k')
    else:
        xlim = [bin_indices[0]*1.1, bin_indices[-1]*1.1]
        plt.xlim(xlim)
        plt.hlines(0, *xlim, colors='k')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    save_spectrum(output_path)
    plt.close()


def plot_signal_and_save(signal, output_path: Path, time=None, ylim=None, extra_command=None):
    samples_count = signal.shape[0]

    plt.figure(figsize=(12, 6))
    if time is not None:
        plt.plot(time, signal, style.color)
        plt.xlabel('time [s]')
        xlim = [time[0], time[-1]]
        plt.xlim(xlim)
    else:
        plt.plot(signal, style.color)
        plt.xlabel('time')
        xlim = [0, samples_count]
        plt.xlim(xlim)
        plt.xticks([])
    plt.yticks([-1, 0, 1])
    plt.ylabel('amplitude')
    plt.hlines(0, xlim[0], xlim[1], 'k')
    if ylim is not None:
        plt.ylim(ylim)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    if extra_command is not None:
        extra_command()

    save_signal(output_path)
    plt.close()


def plot_spectrum_and_save(magnitude_spectrum, output_path: Path, frequencies=None, xticks=None,
                           xtick_labels=None, xlim=None):
    plt.figure(figsize=(12, 6))
    if frequencies is not None:
        plt.plot(frequencies, magnitude_spectrum, style.color)
        if xlim is None:
            xlim = [frequencies[0], frequencies[-1]]
        plt.xlim(xlim)
        plt.xlabel('frequency [Hz]')
        plt.hlines(0, xlim[0], xlim[1], colors='k')
    else:
        plt.plot(magnitude_spectrum, style.color)
        if xlim is None:
            xlim = [0, magnitude_spectrum.shape[0]]
        plt.xlim(xlim)
        plt.xlabel('frequency')
        plt.xticks(xticks, xtick_labels)
        plt.hlines(0, 0, magnitude_spectrum.shape[0], colors='k')
    plt.yticks([])
    plt.ylabel('magnitude')
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    save_spectrum(output_path)
    plt.close()


def plot_spectrum_db_and_save(magnitude_spectrum, output_path: Path, frequencies=None):
    plt.figure(figsize=(12, 6))
    if frequencies is not None:
        plt.plot(frequencies, magnitude_spectrum, style.color)
        plt.xlim([0, frequencies[-1]])
    else:
        plt.plot(magnitude_spectrum, style.color)
        plt.xlim([0, magnitude_spectrum.shape[0]])
    plt.ylim([-60, 0])
    plt.grid()
    plt.xlabel('frequency [Hz]')
    plt.ylabel('magnitude [dBFS]')
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    save_spectrum(output_path)
    plt.close()


def plot_spectrum_db_in_octaves_and_save(magnitude_spectrum, output_path: Path, frequencies):
    plt.figure(figsize=(12, 6))
    min_x = 29
    plt.semilogx(frequencies, magnitude_spectrum, style.color)
    plt.ylim([-60, 0])
    xticks, xtick_labels = frequency_ticks()
    plt.xticks(xticks, xtick_labels)
    plt.xlim([min_x, frequencies[-1]])
    plt.grid()
    plt.xlabel('frequency [Hz]')
    plt.ylabel('magnitude [dBFS]')
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    save_spectrum(output_path)
    plt.close()


def plot_windowed_signal_and_save(signal, window, output_path: Path):
    samples_count = signal.shape[0]

    plt.figure(figsize=(12, 6))
    plt.plot(signal, style.color)
    plt.plot(window, style.window_color)
    plt.yticks([-1, 0, 1])
    plt.xticks([])
    plt.xlim([0, samples_count])
    plt.xlabel('time')
    plt.ylabel('amplitude')
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    save_signal(output_path)
    plt.close()


def plot_window_and_save(window, sampling_rate, output_path):
    samples_count = window.shape[0]
    length_seconds = samples_count / sampling_rate
    time = np.linspace(0, length_seconds, samples_count)
    time -= length_seconds / 2

    plt.figure(figsize=(12, 6))
    plt.plot(time, window, style.color)
    plt.yticks([0, 1])
    time_margin = 0.0
    xlim = [time[0] - time_margin, time[-1] + time_margin]
    plt.xlim(xlim)
    plt.xlabel('time')
    plt.ylabel('amplitude')
    plt.hlines(0, xlim[0], xlim[1], colors='k')
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    save_signal(output_path)
    plt.close()


def plot_analog_signal_and_save(x, y, output_path: Path, xlabel, xticks, xtick_labels, ylim, yticks,
                                ytick_labels):
    plt.figure(figsize=(12, 6))
    plt.plot(x, y, style.color)
    plt.yticks([0, 1])
    xlim = [x[0], x[-1]]
    plt.hlines(0, xlim[0], xlim[1], colors='k')
    plt.hlines(yticks[-1], xlim[0], 0, colors='gray', linestyles='dashed')
    plt.vlines(0, ylim[0], ylim[1], colors='k')
    plt.xlim(xlim)
    plt.xlabel(xlabel)
    plt.xticks(xticks, xtick_labels)
    plt.ylim(ylim)
    plt.yticks(yticks, ytick_labels)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    save_signal(output_path)
    plt.close()
