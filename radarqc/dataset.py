import numpy as np

from radarqc.csfile import CSFile, CSFileHeader
from radarqc.processing import SignalProcessor
from typing import Iterable


class DataSet:
    """Supports aggregation of all Cross-Spectrum files in a given directory
    into a batch of images.

    Uses the monopole antenna channel (Antenna 3) for the spectrum"""

    def __init__(
        self, paths: Iterable[str], preprocess: SignalProcessor
    ) -> None:
        files = (self._load_spectrum(path, preprocess) for path in paths)
        spectra, headers = [], []
        for f in files:
            spectra.append(f.antenna3)
            headers.append(f.header)
        self._spectra = np.stack(spectra)
        self._headers = headers

    @property
    def spectra(self) -> Iterable[np.ndarray]:
        """Image size is (N, num_range, num_doppler), where N is the total
        number of Cross-Spectrum files found in the target directory"""
        return self._spectra

    @property
    def headers(self) -> Iterable[CSFileHeader]:
        """Returns an iterable containing the Cross-Spectrum file header
        for each input path"""
        return self._headers

    def _load_spectrum(self, path: str, preprocess: SignalProcessor) -> CSFile:
        with open(path, "rb") as f:
            return CSFile.load_from(f, preprocess)
