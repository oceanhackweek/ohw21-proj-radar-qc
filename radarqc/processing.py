import abc
import numpy as np


class SignalProcessor(abc.ABC):
    def __call__(self, signal: np.ndarray) -> np.ndarray:
        return self._process(signal)

    @abc.abstractmethod
    def _process(self, signal: np.ndarray) -> np.ndarray:
        """Subclasses will override this functionality"""


class GainCalculator(SignalProcessor):
    def __init__(self, offset: float):
        self._offset = offset

    def _process(self, signal: np.ndarray) -> np.ndarray:
        return 10 * np.log(signal) + self._offset


class Rectifier(SignalProcessor):
    def _process(self, signal: np.ndarray) -> np.ndarray:
        return signal.clip(min=0)


class Abs(SignalProcessor):
    def _process(self, signal: np.ndarray) -> np.ndarray:
        return np.abs(signal)


class CompositeProcessor(SignalProcessor):
    def __init__(self, *processors) -> None:
        self._processors = [p for p in processors]

    def _process(self, signal: np.ndarray) -> np.ndarray:
        for process in self._processors:
            signal = process(signal)
        return signal
