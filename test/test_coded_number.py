from flac.coded_number import encode, decode


def encode_(x: int):
    return int.from_bytes(encode(x), byteorder='big')


def decode_(x: int, n: int):
    return decode(x.to_bytes(n, byteorder='big'))


def test_encode():
    # Single byte values
    assert encode_(0b00000000) == 0b00000000
    assert encode_(0b00011111) == 0b00011111
    assert encode_(0b01111111) == 0b01111111

    # Two bytes values (values between 8 and 11 bits)
    assert encode_(0b10000000) == 0b11000010_10000000
    assert encode_(0b10101010_1) == 0b11000101_10010101
    assert encode_(0b10101010_101) == 0b11010101_10010101


def test_decode():
    # Single byte values
    assert decode_(0b00000000, 1) == 0b00000000
    assert decode_(0b00011111, 1) == 0b00011111
    assert decode_(0b01111111, 1) == 0b01111111

    # Two bytes values (values between 8 and 11 bits)
    assert decode_(0b11000010_10000000, 2) == 0b10000000
    assert decode_(0b11000101_10010101, 2) == 0b10101010_1
    assert decode_(0b11010101_10010101, 2) == 0b10101010_101
