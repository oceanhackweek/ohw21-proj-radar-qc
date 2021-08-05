import datetime
import numpy as np
import pprint
import struct

from typing import BinaryIO, List, Optional, Tuple
from radarqc.serialization import Deserializer, ByteOrder


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
    def __init__(self) -> None:
        self.antenna1 = []
        self.antenna2 = []
        self.antenna3 = []
        self.cross12 = []
        self.cross13 = []
        self.cross23 = []
        self.quality = []

    def to_numpy(self):
        return self._stack(self.antenna1).clip(min=0)

    def _stack(self, antenna):
        return np.stack(antenna)


class _CSFileLoader:
    def load(self, f: BinaryIO) -> Tuple[CSFileHeader, Spectrum]:
        return self._unpack_cs_buffer(f.read())

    def _parse_timestamp(self, seconds: int) -> datetime.datetime:
        start = datetime.datetime(year=1904, month=1, day=1)
        delta = datetime.timedelta(seconds=seconds)
        return start + delta

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
        spectrum = Spectrum()
        for _ in range(header.num_range_cells):
            spectrum.antenna1.append(unpacker.unpack_float(row_size_bytes))
            spectrum.antenna2.append(unpacker.unpack_float(row_size_bytes))
            spectrum.antenna3.append(unpacker.unpack_float(row_size_bytes))
            spectrum.cross12.append(unpacker.unpack_float(row_size_bytes * 2))
            spectrum.cross13.append(unpacker.unpack_float(row_size_bytes * 2))
            spectrum.cross23.append(unpacker.unpack_float(row_size_bytes * 2))
            if header.cskind >= 2:
                spectrum.quality.append(unpacker.unpack_float(row_size_bytes))
        return header, spectrum


class CSFile:
    @staticmethod
    def load_from(f: BinaryIO):
        header, spectrum = _CSFileLoader().load(f)
        return CSFile(header=header, spectrum=spectrum)

    def __init__(self, header: CSFileHeader, spectrum: Spectrum):
        self.header = header
        self.spectrum = spectrum
