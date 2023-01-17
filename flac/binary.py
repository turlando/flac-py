from io import BytesIO, BufferedIOBase


# -----------------------------------------------------------------------------

def mask(n: int) -> int:
    """
    >>> bin(mask(0))
    '0b0'
    >>> bin(mask(1))
    '0b1'
    >>> bin(mask(2))
    '0b11'
    >>> bin(mask(3))
    '0b111'
    """
    return (1 << n) - 1


def extract(x: int, size: int, start: int, stop: int) -> int:
    """
    >>> bin(extract(0b1, 1, 0, 1))
    '0b1'
    >>> bin(extract(0b10101010, 8, 0, 8))
    '0b10101010'
    >>> bin(extract(0b10101010, 8, 2, 5))
    '0b101'
    """
    # assert(0 <= start < stop <= x.bit_length())
    return (x >> (size - stop)) & mask(stop - start)


# -----------------------------------------------------------------------------

def read(buffer: BufferedIOBase, n: int) -> bytes:
    bs = buffer.read(n)
    if n < 1:
        raise ValueError("n must be greater than zero.")
    if bs == b'' or len(bs) != n:
        raise EOFError()
    return bs


def read1(buffer: BufferedIOBase) -> int:
    return read(buffer, 1)[0]


def write1(buffer: BufferedIOBase, x: int) -> int:
    assert 0 <= x < 256
    return buffer.write(x.to_bytes(1, byteorder='big'))


# -----------------------------------------------------------------------------

class _Binary:
    def __init__(self):
        self._bit_offset: int = 0

    @property
    def is_aligned(self) -> bool:
        "Return True if the current offset is at the byte boundary."
        return self._bit_offset == 0

    @property
    def bit_offset(self) -> int:
        "Return the current bit offset"
        return self._bit_offset

    @property
    def bits_until_alignment(self) -> int:
        "Return the number of bits until the offset is at the byte boundary."
        return (8 - self._bit_offset) % 8

    def _add_to_bit_offset(self, n: int):
        self._bit_offset = (self._bit_offset + n) % 8


class Get(_Binary):
    def __init__(self, buffer: BufferedIOBase):
        super().__init__()
        self._buffer = buffer
        self._current_byte = 0

    def _read1(self) -> int:
        b = read1(self._buffer)
        self._current_byte = b
        return b

    def _read(self, n: int) -> bytes:
        return read(self._buffer, n)

    def _read1_if_aligned(self) -> int:
        if self.is_aligned is True:
            return self._read1()
        return self._current_byte

    def uint(self, n: int) -> int:
        if n == 0:
            return 0

        # If all bits that must be read are in the same byte.
        if n <= 8 - self._bit_offset:
            offset = self.bit_offset
            buffer = self._read1_if_aligned()
            self._add_to_bit_offset(n)
            return extract(buffer, 8, offset, offset + n)

        # If the bits are spanning across the byte boundary.
        if n <= 8:
            offset = self.bit_offset
            buffer0 = self._read1_if_aligned()
            buffer1 = self._read1()
            self._add_to_bit_offset(n)
            w = (buffer0 << 8) | buffer1
            return extract(w, 16, offset, offset + n)

        # If the required bits are more than 8.
        #
        # Recursive implementation that is in my opinion clearer but
        # not efficient in Python.
        #
        # > return (self.read(8) << (n - 8)) | self.read(n - 8)
        res = 0
        while n > 8:
            res |= self.uint(8) << (n - 8)
            n -= 8
        return res | self.uint(n)

    def sint(self, n: int):
        x = self.uint(n)
        return x - ((x >> (n - 1)) << n)

    def bool(self) -> bool:
        if self.uint(1) == 1:
            return True
        else:
            return False

    def bytes(self, n: int) -> bytes:
        assert self._bit_offset == 0
        return self._read(n)


class Put(_Binary):
    def __init__(
            self,
    ):
        super().__init__()
        self._buffer = BytesIO()
        self._current_byte = 0

    def _write1(self, x: int):
        return write1(self._buffer, x)

    def _write(self, bs: bytes):
        return self._buffer.write(bs)

    def _flush_if_aligned(self):
        if self.is_aligned is True:
            self._write1(self._current_byte)
            self._current_byte = 0

    @property
    def buffer(self) -> bytes:
        assert self.is_aligned is True
        return self._buffer.getvalue()

    def uint(self, x: int, n: int):
        if n == 0:
            return

        x_ = extract(x, n, 0, n)

        # If a single aligned byte write must be performed.
        if n == 8 and self.is_aligned is True:
            self._write1(x_)
            return

        # If the bits to be written can be stored in the current byte even if
        # the write is not aligned.
        if n <= 8 - self._bit_offset:
            self._current_byte |= x_ << (8 - self.bit_offset - n)
            self._add_to_bit_offset(n)
            self._flush_if_aligned()
            return

        # If the bits are spanning across the byte boundary.
        if n <= 8:
            bits_in_b0 = self.bits_until_alignment
            bits_in_b1 = n - bits_in_b0

            in_b0 = extract(x, n, 0, bits_in_b0)
            in_b1 = extract(x, n, bits_in_b0, n)

            self.uint(in_b0, bits_in_b0)
            self.uint(in_b1, bits_in_b1)
            return

        # If the bits to be written are more than 8.
        m = n
        while m > 8:
            b = extract(x, n, n - m, n - m + 8)
            self.uint(b, 8)
            m -= 8
        b = extract(x, n, n - m, n)
        self.uint(b, m)

    def bool(self, x: bool):
        if x is True:
            self.uint(1, 1)
        else:
            self.uint(0, 1)

    def bytes(self, bs: bytes):
        assert self._bit_offset == 0
        return self._write(bs)
