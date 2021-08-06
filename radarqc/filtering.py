import abc
import numpy as np
from scipy import signal as sig

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class SpectrumFilter(abc.ABC):
    def __call__(self, spectrum: np.ndarray) -> np.ndarray:
        return self._filter(spectrum)

    @abc.abstractmethod
    def _filter(self, spectrum: np.ndarray) -> np.ndarray:
        """Subclasses override to provide custom filtering"""


class NoiseFilter(SpectrumFilter):
    """Computes average across range dimension, and uses a threshold
    to zero out noise regions"""

    def __init__(self, threshold: float, window_std: float) -> None:
        self._threshold = threshold
        self._window_std = window_std

    def _filter(self, spectrum: np.ndarray) -> np.ndarray:
        average = spectrum.mean(axis=0)
        mask = np.where(average < self._threshold, 0, 1)
        length = spectrum.shape[-1]
        window = sig.windows.gaussian(M=length, std=self._window_std * length)
        mask = sig.convolve(mask, window, mode="same")
        mask = (mask - mask.min()) / (mask.max() - mask.min())
        return mask * spectrum


class PreFitPCAFilter(SpectrumFilter):
    def __init__(self, spectra, num_components: int) -> None:
        self._scaler = StandardScaler()
        self._pca = PCA(num_components)
        length = spectra.shape[-1]
        features = spectra.reshape((-1, length))
        features = self._scaler.fit_transform(features)
        self._pca.fit(features)

    def _filter(self, spectrum: np.ndarray) -> np.ndarray:
        features = self._scaler.transform(spectrum)
        transformed = self._pca.transform(features)
        filtered = self._pca.inverse_transform(transformed)
        filtered = self._scaler.inverse_transform(filtered)
        return filtered


class PCAFilter(SpectrumFilter):
    def __init__(self, num_components: int) -> None:
        self._num_components = num_components

    def _filter(self, spectrum: np.ndarray) -> np.ndarray:
        pca = PCA(self._num_components)
        scaler = StandardScaler()
        feature = scaler.fit_transform(spectrum)
        transformed = pca.fit_transform(feature)
        filtered = pca.inverse_transform(transformed)
        filtered = scaler.inverse_transform(filtered)
        return filtered
