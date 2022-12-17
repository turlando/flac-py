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

    def _maybe_read_byte(self):
        if self._bit_offset == 0:
            self._current_byte = self._input.read(1)[0]
        return self._current_byte

    def _update_bit_offset(self, n: int):
        self._bit_offset = (self._bit_offset + n) % 8

    def read(self, n: int):
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
            buffer1 = self._input.read(1)[0]
            offset = self._bit_offset

            self._current_byte = buffer1
            self._update_bit_offset(n)

            w = (buffer0 << 8) | buffer1
            return extract(w, 16, offset, offset + n)

        # If the required bits are more than 8.
        else:
            return (self.read(8) << (n - 8)) | self.read(n - 8)
