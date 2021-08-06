import struct
import enum

from typing import Any


class ByteOrder(enum.Enum):
    BIG_ENDIAN = 1
    LITTLE_ENDIAN = 2
    NETWORK = 3
    NATIVE = 4


class Deserializer:
    _BYTE_ORDER_CHAR = {
        ByteOrder.BIG_ENDIAN: ">",
        ByteOrder.LITTLE_ENDIAN: "<",
        ByteOrder.NETWORK: "!",
        ByteOrder.NATIVE: "=",
    }

    def __init__(self, buff: bytes, byteorder: ByteOrder) -> None:
        self._buff = buff
        self._offset = 0
        self._byteorder = self._BYTE_ORDER_CHAR[byteorder]

    def reset(self) -> None:
        self._offset = 0

    def unpack_char(self, n=1) -> bytes:
        return self._unpack("c", size=1, n=n)

    def unpack_string(self, n=1) -> str:
        raw = self.unpack_char(n=n)
        if n == 1:
            return raw.decode()
        else:
            return "".join(r.decode() for r in raw)

    def unpack_bytes(self, n=1) -> bytes:
        raw = self.unpack_char(n=n)
        if n == 1:
            return raw
        else:
            return b"".join(raw)

    def unpack_int8(self, n=1) -> int:
        return self._unpack(n * "b", size=1, n=n)

    def unpack_uint8(self, n=1) -> int:
        return self._unpack(n * "B", size=1, n=n)

    def unpack_bool(self, n=1) -> bool:
        return self._unpack(n * "?", size=1, n=n)

    def unpack_int16(self, n=1) -> int:
        return self._unpack("h", size=2, n=n)

    def unpack_uint16(self, n=1) -> int:
        return self._unpack("H", size=2, n=n)

    def unpack_int32(self, n=1) -> int:
        return self._unpack("i", size=4, n=n)

    def unpack_uint32(self, n=1) -> int:
        return self._unpack("I", size=4, n=n)

    def unpack_int64(self, n=1) -> int:
        return self._unpack("q", size=8, n=n)

    def unpack_uint64(self, n=1) -> int:
        return self._unpack("Q", size=8, n=n)

    def unpack_float(self, n=1) -> float:
        return self._unpack("f", size=4, n=n)

    def unpack_double(self, n=1) -> float:
        return self._unpack("d", size=8, n=n)

    def _unpack(self, fmt: str, size: int, n: int) -> Any:
        full_fmt = self._build_format(n * fmt)
        data = struct.unpack_from(full_fmt, self._buff, self._offset)
        self._offset += size * n
        if n == 1:
            return data[0]
        else:
            return data

    def _build_format(self, fmt) -> str:
        return "{}{}".format(self._byteorder, fmt)
