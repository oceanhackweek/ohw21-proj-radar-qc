import numpy as np

from typing import List

from radarqc.processing import SignalProcessor


class Spectrum:
    """Stores antenna spectra from Cross-Spectrum files."""

    def __init__(
        self,
        antenna1: np.ndarray,
        antenna2: np.ndarray,
        antenna3: np.ndarray,
        cross12: np.ndarray,
        cross13: np.ndarray,
        cross23: np.ndarray,
        quality: np.ndarray,
        preprocess: SignalProcessor,
    ) -> None:

        self.antenna1 = self._create_real_signal(antenna1, preprocess)
        self.antenna2 = self._create_real_signal(antenna2, preprocess)
        self.antenna3 = self._create_real_signal(antenna3, preprocess)
        self.cross12 = self._create_complex_signal(cross12, preprocess)
        self.cross13 = self._create_complex_signal(cross13, preprocess)
        self.cross23 = self._create_complex_signal(cross23, preprocess)
        self.quality = self._create_real_signal(quality, preprocess)

    def _create_real_signal(
        self, raw: np.ndarray, preprocess: SignalProcessor
    ) -> None:
        return preprocess(raw)

    def _create_complex_signal(
        self, raw: np.ndarray, preprocess: SignalProcessor
    ) -> None:
        real = preprocess(raw.real)
        imag = preprocess(raw.imag)
        return real + 1j * imag
