from dataclasses import dataclass
from enum import Enum
from functools import reduce

from flac.reader import Reader, extract, mask


###############################################################################

MAGIC = int.from_bytes(b'fLaC', byteorder='big')


###############################################################################

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


###############################################################################

class BlockingStrategy(Enum):
    Fixed = 0
    Variable = 1


class BlockSize:
    def __new__(cls, x: int):
        assert 0b0000 < x < 0b1111

        match x:
            case 0b0001:
                return cls.Value(192)
            case n if 0b0010 <= n <= 0b0101:
                return cls.Value(144 * (2 ** n))
            case 0b0110:
                return cls.Uncommon8()
            case 0b0111:
                return cls.Uncommon16()
            case n if 0b1000 <= n <= 0b1111:
                return cls.Value(2 ** n)

    @dataclass(frozen=True)
    class Value:
        size: int

    class Uncommon8:
        pass

    class Uncommon16:
        pass


class SampleRate:
    def __new__(cls, x: int):
        assert 0b0000 <= x < 0b1111

        match x:
            case 0b0000:
                return cls.FromStreaminfo()
            case n if 0b0001 <= n <= 0b1011:
                return cls.Value(n)
            case 0b1100:
                return cls.Uncommon8()
            case 0b1101:
                return cls.Uncommon16()
            case 0b1110:
                return cls.Uncommon16_10()

    class Value(Enum):
        V_88_2_kHz = 0b0001
        V_176_4_kHz = 0b0010
        V_192_kHz = 0b0011
        V_8_kHz = 0b0100
        V_16_kHz = 0b0101
        V_22_05_kHz = 0b0110
        V_24_kHz = 0b0111
        V_32_kHz = 0b1000
        V_44_1_kHz = 0b1001
        V_48_kHz = 0b1010
        V_96_kHz = 0b1100

    class FromStreaminfo:
        pass

    class Uncommon8:
        pass

    class Uncommon16:
        pass

    class Uncommon16_10:
        pass


class Channels(Enum):
    # assert 0b0000 <= x <= 1010
    M = 0b0000
    L_R = 0b0001
    L_R_C = 0b0010
    FL_FR_BL_BR = 0b0011
    FL_FR_FC_BL_BR = 0b0100
    FL_FR_FC_LFE_BL_BR = 0b0101
    FL_FR_FC_LFE_BC_SL_SR = 0b0110
    FL_FR_FC_LFE_BL_BR_SL_SR = 0b0111
    L_S = 0b1000
    S_R = 0b1001
    M_S = 0b1010


class SampleSize:
    def __new__(cls, x: int):
        assert 0b000 <= x <= 0b111
        assert x != 0b011

        match x:
            case 0b000:
                return cls.FromStreaminfo()
            case n if 0b001 <= n <= 0b111:
                return cls.Value(n)

    class FromStreaminfo:
        pass

    class Value(Enum):
        V_8 = 0b001
        V_12 = 0b010
        V_16 = 0b100
        V_20 = 0b101
        V_24 = 0b110
        V_32 = 0b111


@dataclass
class FrameHeader:
    blocking_strategy: BlockingStrategy
    block_size: BlockSize
    sample_rate: SampleRate
    channels: Channels
    sample_size: SampleSize
    coded_number: int
    crc: int


###############################################################################

def decode(reader: Reader):
    assert reader.read(4 * 8) == MAGIC

    streaminfo_header = decode_metadata_block_header(reader)
    assert streaminfo_header.type == MetadataBlockType.Streaminfo
    print(streaminfo_header)

    streaminfo = decode_metadata_block_streaminfo(reader)
    print(streaminfo)

    if streaminfo_header.last is False:
        skip_metadata(reader)

    frame_header = decode_frame_header(reader)
    print(frame_header)


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


def decode_frame_header(reader: Reader):
    assert reader.read(14) == 0b11111111111110
    assert reader.read(1) == 0b0

    blocking_strategy = BlockingStrategy(reader.read(1))
    block_size = BlockSize(reader.read(4))
    sample_rate = SampleRate(reader.read(4))
    channels = Channels(reader.read(4))
    sample_size = SampleSize(reader.read(3))
    assert reader.read(1) == 0b0
    coded_number = decode_coded_number(reader)

    match block_size:
        case BlockSize.Uncommon8:
            block_size_ = reader.read(8)
        case BlockSize.Uncommon16:
            block_size_ = reader.read(16)

    match sample_rate:
        case SampleRate.Uncommon8:
            sample_rate_ = reader.read(8)
        case SampleRate.Uncommon16:
            sample_rate_ = reader.read(16)
        case SampleRate.Uncommon16_10:
            sample_rate_ = reader.read(16) * 10

    crc = reader.read(8)

    return FrameHeader(
        blocking_strategy,
        block_size,
        sample_rate,
        channels,
        sample_size,
        coded_number,
        crc
    )


def _decode_coded_number_reduce(acc: int, x: int):
    return (acc << 6) | mask(6)


def _decode_coded_number_remaining(b0: int):
    if b0 >= 0b11111110:
        return 6
    if b0 >= 0b11111100:
        return 5
    if b0 >= 0b11111000:
        return 4
    if b0 >= 0b11110000:
        return 3
    if b0 >= 0b11100000:
        return 2
    if b0 >= 0b11000000:
        return 1
    else:
        return 0


def decode_coded_number(reader: Reader):
    b0 = reader.read(8)
    r = _decode_coded_number_remaining(b0)
    bs = reader.read_bytes(r)

    b0_ = extract(b0, 8, r + 2, 8)
    bs_ = reduce(_decode_coded_number_reduce, bs, 0)

    return (b0_ << r * 6) | bs_
