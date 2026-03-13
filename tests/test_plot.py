import unittest
import numpy as np
from matplotlib import pyplot as plt

from dspyplot.constants import img_output_path
from dspyplot.plot import stem_signal_and_save, stem_signal

IMG_OUTPUT_PATH = img_output_path("tests")


class TestStem(unittest.TestCase):
    def test_stems_without_xticks(self):
        signal = np.sin(2 * np.pi * np.arange(10) / 5)
        stem_signal_and_save(
            signal, output_path=IMG_OUTPUT_PATH / "stem_sine_10_samples"
        )
        stem_signal(signal)
        ax = plt.gca()
        self.assertEqual(len(ax.get_xticks()), 0)


if __name__ == "__main__":
    unittest.main()
