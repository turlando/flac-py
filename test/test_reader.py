from io import BytesIO
from flac.reader import Reader


###############################################################################

TEST_1 = BytesIO(
    (0b_11010010_00100001_00000100_00001000_00001000_00000111)
    .to_bytes(6, byteorder='big')
)


def test_1():
    r = Reader(TEST_1)
    assert bin(r.read(1)) == '0b1'
    assert bin(r.read(2)) == '0b10'
    assert bin(r.read(3)) == '0b100'
    assert bin(r.read(4)) == '0b1000'
    assert bin(r.read(5)) == '0b10000'
    assert bin(r.read(6)) == '0b100000'
    assert bin(r.read(7)) == '0b1000000'
    assert bin(r.read(8)) == '0b10000000'
    assert bin(r.read(9)) == '0b100000000'
    assert bin(r.read(3)) == '0b111'


###############################################################################

TEST_2 = BytesIO(
    (0b_10000000_00000001)
    .to_bytes(2, byteorder='big')
)


def test_2():
    r = Reader(TEST_2)
    assert bin(r.read(16)) == '0b1000000000000001'


###############################################################################

TEST_3 = BytesIO(
    (0b_00010000_00000000_00000000_00001111)
    .to_bytes(4, byteorder='big')
)


def test_3():
    r = Reader(TEST_3)
    assert bin(r.read(3)) == '0b0'
    assert bin(r.read(25)) == '0b1000000000000000000000000'
    assert bin(r.read(4)) == '0b1111'


###############################################################################

FLAC_MAGIC = int.from_bytes(b'fLaC', byteorder='big')
TEST_FLAC_MAGIC = BytesIO(FLAC_MAGIC.to_bytes(4, byteorder='big'))


def test_flac_magic():
    r = Reader(TEST_FLAC_MAGIC)
    assert r.read(4 * 8) == FLAC_MAGIC
