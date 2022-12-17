from dataclasses import dataclass
from enum import Enum
from flac.reader import Reader


MAGIC = int.from_bytes(b'fLaC', byteorder='big')


class MetadataBlockType(Enum):
    Streaminfo = 0
    Padding = 1
    Application = 2
    Seektable = 3
    VorbisComment = 4
    Cuesheet = 5
    Picture = 6


@dataclass(frozen=True)
class MetadataBlockHeader:
    last: bool
    type: MetadataBlockType
    length: int


@dataclass(frozen=True)
class Streaminfo:
    min_block_size: int
    max_block_size: int
    min_frame_size: int
    max_frame_size: int
    sample_rate: int
    channels: int
    depth: int
    samples: int
    md5: bytes


def decode(reader: Reader):
    assert reader.read(4 * 8) == MAGIC

    streaminfo_header = decode_metadata_block_header(reader)
    assert streaminfo_header.type == MetadataBlockType.Streaminfo
    print(streaminfo_header)

    streaminfo = decode_metadata_block_streaminfo(reader)
    print(streaminfo)

    if streaminfo_header.last is False:
        skip_metadata(reader)


def decode_metadata_block_header(reader: Reader) -> MetadataBlockHeader:
    last = reader.read_bool()
    block_type = MetadataBlockType(reader.read(7))
    length = reader.read(24)
    return MetadataBlockHeader(last, block_type, length)


def decode_metadata_block_streaminfo(reader: Reader) -> Streaminfo:
    min_block = reader.read(16)
    max_block = reader.read(16)
    min_frame = reader.read(24)
    max_frame = reader.read(24)
    sample_rate = reader.read(20)
    channels = reader.read(3) + 1
    depth = reader.read(5) + 1
    samples = reader.read(36)
    md5 = reader.read_bytes(16)

    return Streaminfo(
        min_block, max_block,
        min_frame, max_frame,
        sample_rate, channels, depth,
        samples, md5
    )


def skip_metadata(reader: Reader):
    while True:
        header = decode_metadata_block_header(reader)
        reader._input.read(header.length)
        if header.last is True:
            break
