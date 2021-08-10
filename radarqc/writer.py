import abc
import io
import datetime
import numpy as np
import struct

from collections import defaultdict
from typing import Any, BinaryIO

from radarqc.header import CSFileHeader
from radarqc.serialization import BinaryWriter, ByteOrder
from radarqc.spectrum import Spectrum


class _CSBlockWriter(abc.ABC):
    def write_block(self, writer: BinaryWriter, block: Any) -> None:
        return self._write_block(writer, block)

    @abc.abstractmethod
    def _write_block(self, writer: BinaryWriter, block: Any) -> None:
        """Subclasses will represent different blocks"""


class _RawBlockWriter(_CSBlockWriter):
    def _write_block(self, writer: BinaryWriter, block: bytes) -> None:
        return writer.write_bytes(block)


class CSFileWriter:
    """Responsible for parsing binary data encoded in Cross-Spectrum files"""

    # TODO: build block parsing map
    _BLOCK_WRITERS = defaultdict(_RawBlockWriter)
    _V1_SIZE = 10
    _V2_SIZE = 16
    _V3_SIZE = 24
    _V4_SIZE = 72
    _V5_SIZE = 100

    def dump(
        self, header: CSFileHeader, spectrum: Spectrum, f: BinaryIO
    ) -> None:
        self._write_cs_buff(header, spectrum, f)

    def _write_cs_buff(
        self, header: CSFileHeader, spectrum: Spectrum, f: BinaryIO
    ) -> None:
        writers = {6: self._write_buff_v6}
        version = header.version
        pack = writers[version]
        pack(header, spectrum, f)

    def _write_version(self, f: BinaryIO) -> int:
        int_size_bytes = 4
        buff = f.write(int_size_bytes)
        f.seek(-int_size_bytes, io.SEEK_CUR)
        (version,) = struct.pack_from(">h", buff)
        return version

    def _get_block_parser(self, block_key: str) -> _CSBlockWriter:
        return self._BLOCK_WRITERS[block_key]

    def _get_raw_timestamp(self, timestamp: datetime.datetime) -> int:
        start = datetime.datetime(year=1904, month=1, day=1)
        delta = timestamp - start
        return int(delta.total_seconds())

    def _calculate_section_size_v6(self, blocks: dict) -> int:
        section_size = 0
        for block in blocks.values():
            section_size += len(block)
            section_size += 8
        return section_size

    def _calculate_v1_extent(self, header_size: int) -> int:
        return header_size - self._V1_SIZE

    def _calculate_v2_extent(self, header_size: int) -> int:
        return header_size - self._V2_SIZE

    def _calculate_v3_extent(self, header_size: int) -> int:
        return header_size - self._V3_SIZE

    def _calculate_v4_extent(self, header_size: int) -> int:
        return header_size - self._V4_SIZE

    def _calculate_v5_extent(self, header_size: int) -> int:
        return header_size - self._V5_SIZE

    def _calculate_header_size_v6(self, header: CSFileHeader) -> int:
        section_size_v6 = self._calculate_section_size_v6(header)
        return self._V5_SIZE + section_size_v6 + 4

    def _serialize_blocks(self, header: CSFileHeader) -> dict:
        raw_blocks = {}
        for block_key, block in header.blocks.items():
            block_writer = self._BLOCK_WRITERS[block_key]
            blockio = io.BytesIO()
            writer = BinaryWriter(blockio, ByteOrder.BIG_ENDIAN)
            block_writer.write_block(writer, block)
            raw_blocks[block_key] = blockio.getbuffer()
        return raw_blocks

    def _write_buff_v6(
        self, header: CSFileHeader, spectrum: Spectrum, f: BinaryIO
    ) -> None:
        writer = BinaryWriter(f, ByteOrder.BIG_ENDIAN)
        self._write_header_v6(header, writer)
        self._write_spectrum_data(header, spectrum, writer)

    def _write_header_v6(
        self, header: CSFileHeader, writer: BinaryWriter
    ) -> None:
        blocks = self._serialize_blocks(header)
        header_size = self._calculate_header_size_v6(blocks)

        writer.write_int16(header.version)
        raw_timestamp = self._get_raw_timestamp(header.timestamp)

        writer.write_uint32(raw_timestamp)
        v1_extent = self._calculate_v1_extent(header_size)
        writer.write_int32(v1_extent)  # v1_extent
        # end v1

        writer.write_int16(header.cskind)
        v2_extent = self._calculate_v2_extent(header_size)
        writer.write_int32(v2_extent)  # v2_extent
        # end v2

        writer.write_string(header.site_code)
        v3_extent = self._calculate_v3_extent(header_size)
        writer.write_int32(v3_extent)  # v3_extent
        # end v3

        writer.write_int32(header.cover_minutes)
        writer.write_int32(header.deleted_source)
        writer.write_int32(header.override_source)
        writer.write_float(header.start_freq_mhz)
        writer.write_float(header.rep_freq_mhz)
        writer.write_float(header.bandwidth_khz)
        writer.write_int32(header.sweep_up)
        writer.write_int32(header.num_doppler_cells)
        writer.write_int32(header.num_range_cells)
        writer.write_int32(header.first_range_cell)
        writer.write_float(header.range_cell_dist_km)
        v4_extent = self._calculate_v4_extent(header_size)
        writer.write_int32(v4_extent)  # v4_extent
        # end v4

        writer.write_int32(header.output_interval)
        writer.write_string(header.create_type_code)
        writer.write_string(header.creator_version)
        writer.write_int32(header.num_active_channels)
        writer.write_int32(header.num_spectra_channels)
        writer.write_uint32(header.active_channels)
        v5_extent = self._calculate_v5_extent(header_size)
        writer.write_int32(v5_extent)  # v5_extent
        # end v5

        section_size_v6 = self._calculate_section_size_v6(blocks)
        writer.write_uint32(section_size_v6)
        for block_key, block in blocks.items():
            writer.write_string(block_key)
            writer.write_uint32(len(block))
            writer.write_bytes(block)
        # end v6

    def _write_real_spectrum(
        self, spectrum: np.ndarray, writer: BinaryWriter
    ) -> None:
        dtype = np.dtype("float32").newbyteorder(">")
        buff = spectrum.astype(dtype=dtype)
        writer.write_bytes(buff)

    def _write_complex_spectrum(
        self, spectrum: np.ndarray, writer: BinaryWriter
    ) -> None:
        floats = spectrum.view(np.float32).tolist()
        writer.write_float(floats)

    def _write_spectrum_data(
        self, header: CSFileHeader, spectrum: Spectrum, writer: BinaryWriter
    ) -> None:
        for i in range(header.num_range_cells):
            self._write_real_spectrum(spectrum.antenna1[i], writer)
            self._write_real_spectrum(spectrum.antenna2[i], writer)
            self._write_real_spectrum(spectrum.antenna3[i], writer)
            self._write_complex_spectrum(spectrum.cross12[i], writer)
            self._write_complex_spectrum(spectrum.cross13[i], writer)
            self._write_complex_spectrum(spectrum.cross23[i], writer)
            if header.cskind >= 2:
                self._write_real_spectrum(spectrum.quality[i], writer)
