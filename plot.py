from pathlib import Path

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

import style
import scipy.signal as signal


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


def _stem(points, bin_indices=None, alpha=1.0):
    if bin_indices is not None:
        markerline, stemlines, baseline = plt.stem(bin_indices, points, **style.stem_params)
    else:
        markerline, stemlines, baseline = plt.stem(points, **style.stem_params)
    plt.setp(markerline, color=style.color, alpha=alpha)
    plt.setp(stemlines, color=style.color, alpha=alpha)
    plt.setp(baseline, visible=False)


def stem_signal_and_save(signal, output_path: Path, show_xticks=False, yticks=None, xticks=None):
    samples_count = signal.shape[0]

    plt.figure(figsize=(12, 6))
    _stem(signal)
    if yticks is None:
        plt.yticks([-1, 0, 1])
    else:
        plt.yticks(yticks)
        ylim_multiplier = 1.1
        plt.ylim([ylim_multiplier*yticks[0], ylim_multiplier*yticks[-1]])
    if not show_xticks:
        plt.xticks([])
    if xticks is not None:
        plt.xticks(xticks)
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


def stem_signals_and_save(signal1, signal2, output_path: Path, signal1_alpha=1.0,
                          signal2_alpha=1.0):
    samples_count = signal1.shape[0]

    plt.figure(figsize=(12, 6))
    _stem(signal1, alpha=signal1_alpha)
    _stem(signal2, alpha=signal2_alpha)
    plt.yticks([-1, 0, 1])
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
                           xtick_labels=None, xlim=None, extra_command=None):
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

    if extra_command is not None:
        extra_command()

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


def plot_magnitude_response_and_save(b, a, output_path, ylim, yticks=None, db=False):
    """b, a have their meaning from scipy.signal, see scipy.signal.butter"""
    w, h = signal.freqz(b, a)
    magnitude_response = np.abs(h)

    ylabel = 'Magnitude'

    if db:
        magnitude_response = 20 * np.log10(np.maximum(magnitude_response, 1e-6))
        ylabel += ' [dB]'

    plt.figure(figsize=(12, 6))
    plt.plot(w, magnitude_response, style.color)
    plt.ylim(ylim)
    plt.xlabel('Frequency [radians / sample]')
    plt.margins(0, 0.1)
    plt.grid(which='both', axis='both')
    ticks = np.array([0, np.pi/4, np.pi/2, 3 * np.pi / 4, np.pi])
    plt.xticks(ticks,
               ['$0$', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$', r'$\frac{3\pi}{4}$', r'$\pi$'])
    if yticks is not None:
        plt.yticks(yticks)
    plt.ylabel(ylabel)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    save(output_path, '_magnitude_response')
    plt.close()


def plot_phase_response_and_save(b, a, output_path, yticks=None, ytick_labels=None, unwrap=False):
    """b, a have their meaning from scipy.signal, see scipy.signal.butter"""

    w, h = signal.freqz(b, a)
    phase_response = np.angle(h)
    if unwrap:
        phase_response = np.unwrap(phase_response)

    plt.figure(figsize=(12, 6))
    plt.plot(w, phase_response, style.color)
    plt.xlabel('Frequency [radians / sample]')
    plt.ylabel('Angle [radians]')
    plt.margins(0, 0.1)
    plt.grid(which='both', axis='both')
    ticks = np.array([0, np.pi/4, np.pi/2, 3 * np.pi / 4, np.pi])
    plt.xticks(ticks,
               ['$0$', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$', r'$\frac{3\pi}{4}$', r'$\pi$'])

    if not unwrap:
        if yticks is None:
            yticks = np.arange(-np.pi, np.pi+0.1, np.pi/2)

        if ytick_labels is None:
            ytick_labels = [ r'$-\pi$', r'$-\frac{\pi}{2}$', '$0$', r'$\frac{\pi}{2}$', r'$\pi$']

    plt.yticks(yticks, ytick_labels)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    save(output_path, '_phase_response')
    plt.close()


def plot_group_delay_and_save(b, a, output_path):
    """b, a have their meaning from scipy.signal, see scipy.signal.butter"""
    w, gd = signal.group_delay((b, a))

    plt.figure(figsize=(12, 6))
    plt.plot(w, gd, style.color)
    plt.xlabel('Group delay [samples]')
    plt.ylabel('Angle [radians]')
    plt.margins(0, 0.1)
    plt.grid(which='both', axis='both')
    ticks = np.array([0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi])
    plt.xticks(ticks,
               ['$0$', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$', r'$\frac{3\pi}{4}$', r'$\pi$'])
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    save(output_path, '_group_delay')


def pole_zero_plot_and_save(zeros, poles, output_path):
    """zeros and poles should be array-like of real or complex numbers"""
    pole_zero_plot(zeros, poles)
    save(output_path, suffix='_pole_zero_plot')
    plt.close()


def pole_zero_plot(zeros, poles=None):
    if poles is None:
        poles = []

    ax = plot_z_plane()

    for zero in zeros:
        zero = complex(zero)
        circle = Circle((zero.real, zero.imag), 0.04, edgecolor='k', fill=False, linewidth=3)
        ax.add_patch(circle)

    for pole in poles:
        pole = complex(pole)
        ax.scatter(pole.real, pole.imag, s=200, color='k', marker='x', linewidths=3)


def plot_z_plane():
    unit_circle = Circle((0, 0), 1, edgecolor='k', fill=False)
    # Select length of axes and the space between tick labels
    xmin, xmax, ymin, ymax = -0.5, 0.5, -0.5, 0.5
    # Plot points
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.add_patch(unit_circle)
    # Set identical scales for both axes
    ax.set(xlim=(xmin - 1, xmax + 1), ylim=(ymin - 1, ymax + 1), aspect='equal')
    # Set bottom and left spines as x and y axes of coordinate system
    ax.spines['bottom'].set_position('zero')
    ax.spines['left'].set_position('zero')
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Create labels placed at the end of the axes
    ax.set_xlabel(r'$\Re \{z\}$', size=18, labelpad=-10, x=1.03)
    ax.set_ylabel(r'$\Im \{z\}$', size=18, labelpad=-24, y=1.02, rotation=0)
    minor = True
    ticks = np.arange(-1, 2)
    ax.set_xticks(ticks, minor=minor)
    ax.set_yticks(ticks, minor=minor)
    ax.set_xticklabels([], minor=minor)
    ax.set_yticklabels([], minor=minor)
    minor = False
    ax.set_xticks(np.array(ticks), minor=minor)
    ax.set_yticks(np.array([-1, 1]), minor=minor)
    # Draw major and minor grid lines
    ax.grid(which='both', color='grey', linewidth=1, linestyle='-', alpha=0.2)
    # Draw arrows
    arrow_fmt = dict(markersize=10, color='black', clip_on=False)
    ax.plot(1, 0, marker='>', transform=ax.get_yaxis_transform(), **arrow_fmt)
    ax.plot(0, 1, marker='^', transform=ax.get_xaxis_transform(), **arrow_fmt)

    plt.setp(ax.xaxis.get_majorticklabels(), ha="left")
    plt.setp(ax.yaxis.get_majorticklabels(), ha="left")
    offset_axis_ticklabels_by(fig, ax.xaxis, 5 / 72., -5 / 72.)
    offset_axis_ticklabels_by(fig, ax.yaxis, 15 / 72., -15 / 72.)

    return ax


def offset_axis_ticklabels_by(fig, axis, dx, dy):
    offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)

    # apply offset transform to all ticklabels.
    for label in axis.get_majorticklabels():
        label.set_transform(label.get_transform() + offset)
