from flac.reader import Reader


MAGIC = int.from_bytes(b'fLaC', byteorder='big')


def decode(reader: Reader):
    assert reader.read(4 * 8) == MAGIC
