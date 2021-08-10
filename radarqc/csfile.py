from typing import BinaryIO

import numpy as np

from radarqc.header import CSFileHeader
from radarqc.processing import Identity, SignalProcessor
from radarqc.reader import CSFileReader
from radarqc.writer import CSFileWriter
from radarqc.spectrum import Spectrum


class CSFile:
    """Representation of Cross-Spectrum file for storing CODAR HF radar data."""

    def __init__(self, header: CSFileHeader, spectrum: Spectrum) -> None:
        self._header = header
        self._spectrum = spectrum

    @property
    def header(self) -> CSFileHeader:
        """File header, contains all file metadata"""
        return self._header

    @property
    def spectrum(self) -> Spectrum:
        """Spectrum object containing all file metadata"""
        return self._spectrum

    @property
    def antenna1(self) -> np.ndarray:
        """Spectrum from first loop antenna"""
        return self._spectrum.antenna1

    @property
    def antenna2(self) -> np.ndarray:
        """Spectrum from second loop antenna"""
        return self._spectrum.antenna2

    @property
    def antenna3(self) -> np.ndarray:
        """Spectrum from monopole antenna"""
        return self._spectrum.antenna3

    @property
    def cross12(self) -> np.ndarray:
        """Cross-spectrum from antenna 1 & 2."""
        return self._spectrum.cross12

    @property
    def cross13(self) -> np.ndarray:
        """Cross-spectrum from antenna 1 & 3."""
        return self._spectrum.cross13

    @property
    def cross23(self) -> np.ndarray:
        """Cross-spectrum from antenna 2 & 3."""
        return self._spectrum.cross23


def load(f: BinaryIO, preprocess: SignalProcessor = None) -> CSFile:
    if preprocess is None:
        preprocess = Identity()

    header, spectrum = CSFileReader().load(f, preprocess)
    return CSFile(header, spectrum)


def dump(cs: CSFile, f: BinaryIO) -> None:
    header, spectrum = cs.header, cs.spectrum
    CSFileWriter().dump(header, spectrum, f)
