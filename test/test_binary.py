from io import BytesIO
from flac.binary import Get, Put


# -----------------------------------------------------------------------------

TEST_GET_1 = BytesIO(
    (0b_11010010_00100001_00000100_00001000_00001000_00000111)
    .to_bytes(6, byteorder='big')
)


def test_get_1():
    g = Get(TEST_GET_1)
    assert g.uint(1) == 0b1
    assert g.uint(2) == 0b10
    assert g.uint(3) == 0b100
    assert g.uint(4) == 0b1000
    assert g.uint(5) == 0b10000
    assert g.uint(6) == 0b100000
    assert g.uint(7) == 0b1000000
    assert g.uint(8) == 0b10000000
    assert g.uint(9) == 0b100000000
    assert g.uint(3) == 0b111


# -----------------------------------------------------------------------------

TEST_GET_2 = BytesIO(
    (0b_10000000_00000001)
    .to_bytes(2, byteorder='big')
)


def test_get_2():
    g = Get(TEST_GET_2)
    assert g.uint(16) == 0b10000000_00000001


# -----------------------------------------------------------------------------

TEST_GET_3 = BytesIO(
    (0b_00010000_00000000_00000000_00001111)
    .to_bytes(4, byteorder='big')
)


def test_get_3():
    g = Get(TEST_GET_3)
    assert g.uint(3) == 0b000
    assert g.uint(25) == 0b10000_00000000_00000000_0000
    assert g.uint(4) == 0b1111


# -----------------------------------------------------------------------------

FLAC_MAGIC = int.from_bytes(b'fLaC', byteorder='big')
TEST_GET_FLAC_MAGIC = BytesIO(FLAC_MAGIC.to_bytes(4, byteorder='big'))


def test_get_flac_magic():
    g = Get(TEST_GET_FLAC_MAGIC)
    assert g.uint(4 * 8) == FLAC_MAGIC


# -----------------------------------------------------------------------------

def test_put_1():
    p = Put()

    p.uint(0b10101010, 8)
    assert p.buffer[0] == 0b10101010

    p.uint(0b01010101, 8)
    assert p.buffer[1] == 0b1010101


# -----------------------------------------------------------------------------

def test_put_2():
    p = Put()

    p.uint(0b1, 1)
    p.uint(0b01, 2)
    p.uint(0b010, 3)
    p.uint(0b10, 2)

    assert p.buffer[0] == 0b10101010


# -----------------------------------------------------------------------------

def test_put_3():
    p = Put()

    p.uint(0b10000, 5)
    p.uint(0b11111111, 8)
    p.uint(0b001, 3)
    assert p.buffer[0] == 0b10000111
    assert p.buffer[1] == 0b11111001


# -----------------------------------------------------------------------------

def test_put_4():
    p = Put()

    p.uint(0b1000, 4)
    p.uint(0b1111111111111111, 16)
    p.uint(0b0001, 4)

    assert p.buffer[0] == 0b10001111
    assert p.buffer[1] == 0b11111111
    assert p.buffer[2] == 0b11110001


# -----------------------------------------------------------------------------

def test_put_5():
    p = Put()

    # Simulate writing Streaminfo that caused 'ValueError: negative shift
    # count' in extract when writing more than 8 bits.
    p.uint(0, 16)  # bytes 0, 1
    p.uint(0, 16)  # bytes 2, 3
    p.uint(0, 24)  # bytes 4, 5, 6
    p.uint(0, 24)  # bytes 7, 8, 9
    p.uint(0, 20)  # bytes 10, 11

    # Simulate writing Streaminfo that caused 'ValueError: negative shift
    # count' in extract when writing less than 8 bits spanning across the
    # byte boundary.
    p.uint(0b111, 3)  # bytes 11, 12
    p.uint(0b11111, 5)  # bytes 12, 13
    p.uint(0b1, 4)  # byte 13

    assert p.buffer[12] == 0b00001111
    assert p.buffer[13] == 0b11110001

    assert p.is_aligned is True
