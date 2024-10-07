from os import walk
from pathlib import Path
import soundfile as sf
import numpy as np
from .signals import normalize_for_listening, apply_fade, normalize_to_peak
import logging as logger
import pyloudnorm as pyln


def preprocess(signal, sample_rate: float):
    signal = apply_fade(signal, fade_length=1000)
    signal = normalize_for_listening(signal, sample_rate)

    if np.any(np.abs(signal) >= 1):
        logger.warning(
            f"Audio signal would clip after normalization to -16 dB LUFS. Normalizing to -1 dB instead."
        )
        signal = pyln.normalize.peak(signal, -1)

    return signal


def save_audio_file_with_normalization(output_path: Path, signal, sample_rate: float):
    preprocessed_signal = _preprocess(np.copy(signal), sample_rate)

    if np.any(np.abs(preprocessed_signal) >= 1):
        logger.warning(
            f"Audio signal {output_path.stem} meant for listening is probably clipped; it's amplitude is larger or equal to 1 AFTER normalization"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path = output_path.with_suffix(".flac")

    sf.write(output_path, preprocessed_signal, sample_rate)
