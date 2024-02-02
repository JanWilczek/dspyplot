import unittest
from dspyplot import audiofile
from pathlib import Path
import numpy as np


class TestAudiofile(unittest.TestCase):
            def test_save_audio_file_with_normalization(self):
                        sample_rate = 44100
                        sine = np.sin(2 * np.pi * 440 * np.arange(sample_rate * 5) / sample_rate)
                        audiofile.save_audio_file_with_normalization(Path('test_output') / 'sine', sine, sample_rate)
                        
if __name__ == "__main__":
            unittest.main()

