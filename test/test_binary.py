from io import BytesIO
from flac.binary import Reader, Writer


# -----------------------------------------------------------------------------

TEST_READER_1 = BytesIO(
    (0b_11010010_00100001_00000100_00001000_00001000_00000111)
    .to_bytes(6, byteorder='big')
)


def test_reader_1():
    r = Reader(TEST_READER_1)
    assert r.read_uint(1) == 0b1
    assert r.read_uint(2) == 0b10
    assert r.read_uint(3) == 0b100
    assert r.read_uint(4) == 0b1000
    assert r.read_uint(5) == 0b10000
    assert r.read_uint(6) == 0b100000
    assert r.read_uint(7) == 0b1000000
    assert r.read_uint(8) == 0b10000000
    assert r.read_uint(9) == 0b100000000
    assert r.read_uint(3) == 0b111


# -----------------------------------------------------------------------------

TEST_READER_2 = BytesIO(
    (0b_10000000_00000001)
    .to_bytes(2, byteorder='big')
)


def test_reader_2():
    r = Reader(TEST_READER_2)
    assert r.read_uint(16) == 0b10000000_00000001


# -----------------------------------------------------------------------------

TEST_READER_3 = BytesIO(
    (0b_00010000_00000000_00000000_00001111)
    .to_bytes(4, byteorder='big')
)


def test_reader_3():
    r = Reader(TEST_READER_3)
    assert r.read_uint(3) == 0b000
    assert r.read_uint(25) == 0b10000_00000000_00000000_0000
    assert r.read_uint(4) == 0b1111


# -----------------------------------------------------------------------------

FLAC_MAGIC = int.from_bytes(b'fLaC', byteorder='big')
TEST_READER_FLAC_MAGIC = BytesIO(FLAC_MAGIC.to_bytes(4, byteorder='big'))


def test_reader_flac_magic():
    r = Reader(TEST_READER_FLAC_MAGIC)
    assert r.read_uint(4 * 8) == FLAC_MAGIC


# -----------------------------------------------------------------------------

def test_writer_1():
    b = BytesIO()
    w = Writer(b)

    w.write(0b10101010, 8)
    assert b.getvalue()[0] == 0b10101010

    w.write(0b01010101, 8)
    assert b.getvalue()[1] == 0b1010101


# -----------------------------------------------------------------------------

def test_writer_2():
    b = BytesIO()
    w = Writer(b)

    w.write(0b1, 1)
    w.write(0b01, 2)
    w.write(0b010, 3)
    w.write(0b10, 2)
    assert b.getvalue()[0] == 0b10101010


# -----------------------------------------------------------------------------

def test_writer_3():
    b = BytesIO()
    w = Writer(b)

    w.write(0b10000, 5)
    w.write(0b11111111, 8)
    w.write(0b001, 3)
    assert b.getvalue()[0] == 0b10000111
    assert b.getvalue()[1] == 0b11111001


# -----------------------------------------------------------------------------

def test_writer_4():
    b = BytesIO()
    w = Writer(b)

    w.write(0b1000, 4)
    w.write(0b1111111111111111, 16)
    w.write(0b0001, 4)
    assert b.getvalue()[0] == 0b10001111
    assert b.getvalue()[1] == 0b11111111
    assert b.getvalue()[2] == 0b11110001


# -----------------------------------------------------------------------------

def test_writer_5():
    b = BytesIO()
    w = Writer(b)

    # Simulate writing Streaminfo that caused ValueError: negative shift count
    # in extract.
    w.write(0, 16)
    w.write(0, 16)
    w.write(0, 24)
    w.write(0, 24)
    w.write(0, 20)
