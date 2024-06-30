from math import cos, pi

import scipy.signal


def main():
    for window_size in [10, 11]:
        scipy_window = scipy.signal.windows.hann(window_size, False)
        print(scipy_window)
        self_calculated_window = [
            0.5 * (1 - cos(2 * pi * n / window_size)) for n in range(0, window_size)
        ]
        print(self_calculated_window)


if __name__ == "__main__":
    main()
