import pytest
from flac.coded_number import encode, decode


# [(decoded, encoded, size)]
TESTS = [
    # Single byte values
    (0b00000000, 0b00000000, 1),
    (0b00011111, 0b00011111, 1),
    (0b01111111, 0b01111111, 1),

    # Two bytes values (values between 8 and 11 bits)
    (0b10000000, 0b11000010_10000000, 2),
    (0b10101010_1, 0b11000101_10010101, 2),
    (0b10101010_101, 0b11010101_10010101, 2)
]


def encode_(x: int):
    return int.from_bytes(encode(x), byteorder='big')


def decode_(x: int, n: int):
    return decode(x.to_bytes(n, byteorder='big'))


@pytest.mark.parametrize(("decoded", "encoded", "size"), TESTS)
def test(decoded, encoded, size):
    assert encode_(decoded) == encoded
    assert decode_(encoded, size) == decoded
