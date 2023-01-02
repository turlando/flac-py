from io import BytesIO


def mask(n: int):
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


def extract(x: int, size: int, start: int, stop: int):
    """
    >>> bin(extract(0b10101010, 8, 0, 8))
    '0b10101010'
    >>> bin(extract(0b10101010, 8, 2, 5))
    '0b101'
    """
    # assert(0 <= start < stop <= x.bit_length())
    return (x >> (size - stop)) & mask(stop - start)


class Reader:
    def __init__(self, input: BytesIO):
        self._input = input
        self._bit_offset = 0
        self._current_byte = 0

    @property
    def is_byte_aligned(self):
        return self._bit_offset == 0

    @property
    def bits_until_byte_alignment(self):
        return 8 - self._bit_offset

    def _read_byte(self):
        b = self._input.read(1)
        if b == b'':
            raise EOFError()
        return b[0]

    def _read_bytes(self, n: int):
        bs = self._input.read(n)
        if len(bs) != n:
            raise EOFError()
        return bs

    def _maybe_read_byte(self):
        if self._bit_offset == 0:
            self._current_byte = self._read_byte()
        return self._current_byte

    def _update_bit_offset(self, n: int):
        self._bit_offset = (self._bit_offset + n) % 8

    def read_uint(self, n: int):
        if n == 0:
            return 0

        # If all bits that must be read are in the same byte.
        if n <= 8 - self._bit_offset:
            buffer = self._maybe_read_byte()
            offset = self._bit_offset

            self._update_bit_offset(n)

            return extract(buffer, 8, offset, offset + n)

        # If the bits are spanning across the byte boundary.
        if n <= 8:
            buffer0 = self._maybe_read_byte()
            buffer1 = self._read_byte()
            offset = self._bit_offset

            self._current_byte = buffer1
            self._update_bit_offset(n)

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
            res |= self.read_uint(8) << (n - 8)
            n -= 8
        return res | self.read_uint(n)

    def read_int(self, n: int):
        x = self.read_uint(n)
        return x - ((x >> (n - 1)) << n)

    def read_bool(self) -> bool:
        if self.read_uint(1) == 1:
            return True
        else:
            return False

    def read_bytes(self, n: int):
        assert self._bit_offset == 0
        return self._read_bytes(n)


class Writer:
    def __init__(self, output: BytesIO):
        self._output = output
        self._bit_offset = 0
        self._current_byte = 0

    def _update_bit_offset(self, n: int):
        self._bit_offset = (self._bit_offset + n) % 8

    def _maybe_flush(self):
        if self._bit_offset == 0:
            self._output.write(self._current_byte.to_bytes(1, byteorder='big'))
            self._current_byte = 0

    def write(self, x: int, n: int):
        if n == 0:
            return

        x_ = extract(x, n, 0, n)

        # If a single aligned byte write must be performed.
        if n == 8 and self._bit_offset == 0:
            self._output.write(x_.to_bytes(1, byteorder='big'))
            return

        # If the bits to be written are less than a single byte and possibly
        # not aligned.
        if n <= 8 - self._bit_offset:
            self._current_byte |= x_ << (8 - self._bit_offset - n)
            self._update_bit_offset(n)
            self._maybe_flush()
            return

        # If the bits are spanning across the byte boundary.
        if n <= 8:
            bits_in_b0 = n - self._bit_offset
            bits_in_b1 = 8 - bits_in_b0

            in_b0 = extract(x, n, 0, bits_in_b0)
            in_b1 = extract(x, n, bits_in_b0, n)

            self.write(in_b0, bits_in_b0)
            self.write(in_b1, bits_in_b1)
            return

        # If the bits to be written are more than 8.
        m = n
        while m > 8:
            b = extract(x, n, n - m, n - m + 8)
            self.write(b, 8)
            m -= 8
        b = extract(x, n, n - m, n - m + 8)
        self.write(b, m)

    def write_bool(self, x: bool):
        if x is True:
            self.write(1, 1)
        else:
            self.write(0, 1)

    def write_bytes(self, b: bytes):
        assert self._bit_offset == 0
        return self._output.write(b)
