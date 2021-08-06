import abc
import datetime
import numpy as np
import pprint
import struct

from radarqc.processing import Abs, GainCalculator, CompositeProcessor
from radarqc.serialization import Deserializer, ByteOrder

from typing import BinaryIO, List, Tuple


class CSFileHeader:
    def __init__(self):
        self.version = None
        self.timestamp = None
        self.cskind = None
        self.site_code = None
        self.cover_minutes = None
        self.deleted_source = None
        self.override_source = None
        self.start_freq_mhz = None
        self.rep_freq_mhz = None
        self.bandwidth_khz = None
        self.sweep_up = None
        self.num_doppler_cells = None
        self.num_range_cells = None
        self.first_range_cell = None
        self.range_cell_dist_km = None
        self.output_interval = None
        self.create_type_code = None
        self.creator_version = None
        self.num_active_channels = None
        self.num_spectra_channels = None
        self.active_channels = None
        self.blocks = {}

    def __repr__(self) -> str:
        return pprint.pformat(self.__dict__)


class Spectrum:
    _GAIN_OFFSET = -34.2

    def __init__(
        self,
        antenna1: List[float],
        antenna2: List[float],
        antenna3: List[float],
        cross12: List[float],
        cross13: List[float],
        cross23: List[float],
        quality: List[float],
    ) -> None:
        self._preprocess = CompositeProcessor(
            Abs(), GainCalculator(offset=self._GAIN_OFFSET)
        )

        self.antenna1 = self._create_real_signal(antenna1)
        self.antenna2 = self._create_real_signal(antenna2)
        self.antenna3 = self._create_real_signal(antenna3)
        self.cross12 = self._create_complex_signal(cross12)
        self.cross13 = self._create_complex_signal(cross13)
        self.cross23 = self._create_complex_signal(cross23)
        self.quality = self._create_real_signal(quality)

    def _create_real_signal(self, raw: List[float]) -> None:
        return self._preprocess(self._to_numpy(raw))

    def _create_complex_signal(self, raw: List[float]) -> None:
        signal = self._to_numpy(raw)
        real = self._preprocess(signal[:, 0::2])
        imag = self._preprocess(signal[:, 1::2])
        return real + 1j * imag

    def _to_numpy(self, raw: List[float]) -> np.ndarray:
        return np.stack(raw)


class _CSFileLoader:
    def load(self, f: BinaryIO) -> Tuple[CSFileHeader, Spectrum]:
        return self._unpack_cs_buffer(f.read())

    def _parse_timestamp(self, seconds: int) -> datetime.datetime:
        start = datetime.datetime(year=1904, month=1, day=1)
        delta = datetime.timedelta(seconds=seconds)
        return start + delta

    def _parse_key_time(self, buffer: bytes) -> dict:
        unpacker = Deserializer(buffer, ByteOrder.BIG_ENDIAN)
        mark = unpacker.unpack_uint8()
        date = datetime.datetime(
            year=unpacker.unpack_uint16(),
            month=unpacker.unpack_uint8(),
            day=unpacker.unpack_uint8(),
            hour=unpacker.unpack_uint8(),
            minute=unpacker.unpack_uint8(),
            second=unpacker.unpack_double(),
        )
        coverage = unpacker.unpack_double()
        hours_from_utc = unpacker.unpack_double()

        return {
            "mark": mark,
            "date": date,
            "coverage": coverage,
            "hours_from_utc": hours_from_utc,
        }

    def _parse_key_zone(self, buffer: bytes) -> str:
        return buffer.decode()

    def _parse_key_city(self, buffer: bytes) -> str:
        return buffer.decode()

    def _parse_key_loca(self, buffer: bytes) -> dict:
        unpacker = Deserializer(buffer, byteorder=ByteOrder.BIG_ENDIAN)
        latitude = unpacker.unpack_double()
        longitude = unpacker.unpack_double()
        altitude_meters = unpacker.unpack_double()
        return {
            "latitude": latitude,
            "longitude": longitude,
            "altitude_meters": altitude_meters,
        }

    def _parse_key_sitd(self, buffer: bytes) -> str:
        return buffer.decode()

    def _parse_key_rcvi(self, buffer: bytes) -> dict:
        unpacker = Deserializer(buffer, byteorder=ByteOrder.BIG_ENDIAN)
        receiver_model = unpacker.unpack_uint32()
        antenna_model = unpacker.unpack_uint32()
        reference_gain_db = unpacker.unpack_double()
        firmware = unpacker.unpack_string(32)
        return {
            "receiver_model": receiver_model,
            "antenna_model": antenna_model,
            "reference_gain_db": reference_gain_db,
            "firmware": firmware,
        }

    def _parse_key_tool(self, buffer: bytes) -> str:
        return buffer.decode()

    def _parse_key_glrm(self, buffer: bytes) -> dict:
        unpacker = Deserializer(buffer, byteorder=ByteOrder.BIG_ENDIAN)
        method = unpacker.unpack_uint8()
        version = unpacker.unpack_uint8()
        points_removed = unpacker.unpack_uint32()
        times_removed = unpacker.unpack_uint32()
        segments_removed = unpacker.unpack_uint32()
        point_power_thresh = unpacker.unpack_double()
        range_power_thresh = unpacker.unpack_double()
        range_bin_thresh = unpacker.unpack_double()
        remove_dc = bool(unpacker.unpack_uint8())
        return {
            "method": method,
            "version": version,
            "points_removed": points_removed,
            "times_removed": times_removed,
            "segments_removed": segments_removed,
            "point_power_thresh": point_power_thresh,
            "range_power_thresh": range_power_thresh,
            "range_bin_thresh": range_bin_thresh,
            "remove_dc": remove_dc,
        }

    def _parse_key_supi(self, buffer: bytes) -> dict:
        unpacker = Deserializer(buffer, byteorder=ByteOrder.BIG_ENDIAN)
        method = unpacker.unpack_uint8()
        version = unpacker.unpack_uint8()
        mode = unpacker.unpack_uint8()
        debug_mode = unpacker.unpack_uint8()
        doppler_suppressed = unpacker.unpack_uint32()
        power_thresh = unpacker.unpack_double()
        range_bin_thresh = unpacker.unpack_double()
        range_banding = unpacker.unpack_int16()
        detection_smoothing = unpacker.unpack_int16()
        return {
            "method": method,
            "version": version,
            "mode": mode,
            "debug_mode": debug_mode,
            "doppler_suppressed": doppler_suppressed,
            "power_thresh": power_thresh,
            "range_bin_thresh": range_bin_thresh,
            "range_banding": range_banding,
            "detection_smoothing": detection_smoothing,
        }

    def _parse_key_supm(self, buffer: bytes, header: CSFileHeader) -> dict:
        block_size = header.num_spectra_channels * header.num_doppler_cells * 4

    def _unpack_cs_buffer(self, buff: bytes) -> Tuple[CSFileHeader, Spectrum]:
        unpackers = {6: self._unpack_buffer_v6}
        (version,) = struct.unpack_from(">h", buff)
        unpack = unpackers[version]
        return unpack(buff)

    def _unpack_buffer_v6(self, buff: bytes) -> Tuple[CSFileHeader, Spectrum]:
        unpacker = Deserializer(buff, byteorder=ByteOrder.BIG_ENDIAN)
        header = CSFileHeader()

        header.version = unpacker.unpack_int16()
        header.timestamp = self._parse_timestamp(unpacker.unpack_uint32())
        unpacker.unpack_int32()
        # end v1

        header.cskind = unpacker.unpack_int16()
        unpacker.unpack_int32()
        # end v2

        header.site_code = unpacker.unpack_string(4)
        unpacker.unpack_int32()
        # end v3

        header.cover_minutes = unpacker.unpack_int32()
        header.deleted_source = bool(unpacker.unpack_int32())
        header.override_source = bool(unpacker.unpack_int32())
        header.start_freq_mhz = unpacker.unpack_float()
        header.rep_freq_mhz = unpacker.unpack_float()
        header.bandwidth_khz = unpacker.unpack_float()
        header.sweep_up = bool(unpacker.unpack_int32())
        header.num_doppler_cells = unpacker.unpack_int32()
        header.num_range_cells = unpacker.unpack_int32()
        header.first_range_cell = unpacker.unpack_int32()
        header.range_cell_dist_km = unpacker.unpack_float()
        unpacker.unpack_int32()
        # end v4

        header.output_interval = unpacker.unpack_int32()
        header.create_type_code = unpacker.unpack_string(4)
        header.creator_version = unpacker.unpack_string(4)
        header.num_active_channels = unpacker.unpack_int32()
        header.num_spectra_channels = unpacker.unpack_int32()
        header.active_channels = unpacker.unpack_uint32()
        unpacker.unpack_int32()
        # end v5

        cs6_header_size = unpacker.unpack_uint32()
        while cs6_header_size > 0:
            block_key = unpacker.unpack_string(4)
            block_size = unpacker.unpack_uint32()

            block = unpacker.unpack_bytes(n=block_size)
            header.blocks[block_key] = block

            cs6_header_size -= 8
            cs6_header_size -= block_size
        # end v6

        row_size_bytes = header.num_doppler_cells
        a1, a2, a3, c12, c13, c23, q = [], [], [], [], [], [], []
        for _ in range(header.num_range_cells):
            a1.append(unpacker.unpack_float(row_size_bytes))
            a2.append(unpacker.unpack_float(row_size_bytes))
            a3.append(unpacker.unpack_float(row_size_bytes))
            c12.append(unpacker.unpack_float(row_size_bytes * 2))
            c13.append(unpacker.unpack_float(row_size_bytes * 2))
            c23.append(unpacker.unpack_float(row_size_bytes * 2))
            if header.cskind >= 2:
                q.append(unpacker.unpack_float(row_size_bytes))
        spectrum = Spectrum(a1, a2, a3, c12, c13, c23, q)
        return header, spectrum


class CSFile:
    @staticmethod
    def load_from(f: BinaryIO):
        header, spectrum = _CSFileLoader().load(f)
        return CSFile(header=header, spectrum=spectrum)

    def __init__(self, header: CSFileHeader, spectrum: Spectrum) -> None:
        self._header = header
        self._spectrum = spectrum

    @property
    def header(self) -> CSFileHeader:
        return self._header

    @property
    def antenna1(self) -> np.ndarray:
        return self._spectrum.antenna1

    @property
    def antenna2(self) -> np.ndarray:
        return self._spectrum.antenna2

    @property
    def antenna3(self) -> np.ndarray:
        return self._spectrum.antenna3

    @property
    def cross12(self) -> np.ndarray:
        return self._spectrum.cross12

    @property
    def cross13(self) -> np.ndarray:
        return self._spectrum.cross13

    @property
    def cross23(self) -> np.ndarray:
        return self._spectrum.cross23
