from dataclasses import dataclass
from enum import Enum
from functools import reduce
from typing import Optional

from flac.binary import Reader, extract, mask


###############################################################################

MAGIC = int.from_bytes(b'fLaC', byteorder='big')


def consume_magic(reader: Reader):
    assert reader.read_uint(4 * 8) == MAGIC


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


def read_metadata_block_header(reader: Reader) -> MetadataBlockHeader:
    return MetadataBlockHeader(
        last=reader.read_bool(),
        type=MetadataBlockType(reader.read_uint(7)),
        length=reader.read_uint(24)
    )


###############################################################################

@dataclass(frozen=True)
class Streaminfo:
    min_block_size: int
    max_block_size: int
    min_frame_size: int
    max_frame_size: int
    sample_rate: int
    channels: int
    sample_size: int
    samples: int
    md5: bytes


def read_metadata_block_streaminfo(reader: Reader) -> Streaminfo:
    return Streaminfo(
        min_block_size=reader.read_uint(16),
        max_block_size=reader.read_uint(16),
        min_frame_size=reader.read_uint(24),
        max_frame_size=reader.read_uint(24),
        sample_rate=reader.read_uint(20),
        channels=reader.read_uint(3) + 1,
        sample_size=reader.read_uint(5) + 1,
        samples=reader.read_uint(36),
        md5=reader.read_bytes(16)
    )


###############################################################################

def skip_metadata(reader: Reader):
    while True:
        header = read_metadata_block_header(reader)
        reader.read_bytes(header.length)
        if header.last is True:
            break


###############################################################################

class BlockingStrategy(Enum):
    Fixed = 0
    Variable = 1


def read_blocking_strategy(reader: Reader) -> BlockingStrategy:
    return BlockingStrategy(reader.read_uint(1))


###############################################################################

@dataclass(frozen=True)
class BlockSizeValue:
    size: int


@dataclass(frozen=True)
class BlockSizeUncommon8:
    pass


@dataclass(frozen=True)
class BlockSizeUncommon16:
    pass


BlockSize = BlockSizeValue | BlockSizeUncommon8 | BlockSizeUncommon16


def read_block_size(reader: Reader) -> BlockSize:
    x = reader.read_uint(4)
    assert 0b0000 < x < 0b1111

    match x:
        case 0b0001:
            return BlockSizeValue(192)
        case n if 0b0010 <= n <= 0b0101:
            return BlockSizeValue(144 * (2 ** n))
        case 0b0110:
            return BlockSizeUncommon8()
        case 0b0111:
            return BlockSizeUncommon16()
        case n if 0b1000 <= n <= 0b1111:
            return BlockSizeValue(2 ** n)

    raise ValueError(f"Cannot read block size: {bin(x)}")


###############################################################################

class SampleRateFromStreaminfo:
    pass


class SampleRateValue(Enum):
    V_88_2_kHz = 88_200
    V_176_4_kHz = 176_400
    V_192_kHz = 192_000
    V_8_kHz = 8_000
    V_16_kHz = 16_000
    V_22_05_kHz = 22_050
    V_24_kHz = 24_000
    V_32_kHz = 32_000
    V_44_1_kHz = 44_100
    V_48_kHz = 48_000
    V_96_kHz = 96_000

    @classmethod
    def from_bin(cls, x: int):
        return {
            0b0001: cls.V_88_2_kHz,
            0b0010: cls.V_176_4_kHz,
            0b0011: cls.V_192_kHz,
            0b0100: cls.V_8_kHz,
            0b0101: cls.V_16_kHz,
            0b0110: cls.V_22_05_kHz,
            0b0111: cls.V_24_kHz,
            0b1000: cls.V_32_kHz,
            0b1001: cls.V_44_1_kHz,
            0b1010: cls.V_48_kHz,
            0b1100: cls.V_96_kHz
        }[x]

    def to_int(self):
        return self.value


class SampleRateUncommon8:
    pass


class SampleRateUncommon16:
    pass


class SampleRateUncommon16_10:
    pass


SampleRate = (
    SampleRateFromStreaminfo
    | SampleRateValue
    | SampleRateUncommon8
    | SampleRateUncommon16
    | SampleRateUncommon16_10
)


def read_sample_rate(reader: Reader) -> SampleRate:
    x = reader.read_uint(4)
    assert 0b0000 <= x < 0b1111

    match x:
        case 0b0000:
            return SampleRateFromStreaminfo()
        case n if 0b0001 <= n <= 0b1011:
            return SampleRateValue.from_bin(n)
        case 0b1100:
            return SampleRateUncommon8()
        case 0b1101:
            return SampleRateUncommon16()
        case 0b1110:
            return SampleRateUncommon16_10()

    raise ValueError(f"Cannot read sample rate: {bin(x)}")


###############################################################################

class Channels(Enum):
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

    @property
    def count(self) -> int:
        return {
            Channels.M: 1,
            Channels.L_R: 2,
            Channels.L_R_C: 3,
            Channels.FL_FR_BL_BR: 4,
            Channels.FL_FR_FC_BL_BR: 5,
            Channels.FL_FR_FC_LFE_BL_BR: 6,
            Channels.FL_FR_FC_LFE_BC_SL_SR: 7,
            Channels.FL_FR_FC_LFE_BL_BR_SL_SR: 8,
            Channels.L_S: 2,
            Channels.S_R: 2,
            Channels.M_S: 2
        }[self]

    @property
    def decorrelation_bit(self):
        # Side channel has one extra bit in sample_size
        match self:
            case self.L_S:
                return (0, 1)
            case self.S_R:
                return (1, 0)
            case self.M_S:
                return (0, 1)
            case _:
                return (0,) * self.count


def read_channels(reader: Reader) -> Channels:
    x = reader.read_uint(4)
    assert 0b0000 <= x <= 0b1010
    return Channels(x)


###############################################################################

@dataclass(frozen=True)
class SampleSizeFromStreaminfo:
    pass


class SampleSizeValue(Enum):
    V_8 = 8
    V_12 = 12
    V_16 = 16
    V_20 = 20
    V_24 = 24
    V_32 = 32

    @classmethod
    def from_bin(cls, x: int):
        return {
            0b001: cls.V_8,
            0b010: cls.V_12,
            0b100: cls.V_16,
            0b101: cls.V_20,
            0b110: cls.V_24,
            0b111: cls.V_32
        }[x]

    def to_int(self):
        return self.value


SampleSize = SampleSizeFromStreaminfo | SampleSizeValue


def read_sample_size(reader: Reader) -> SampleSize:
    x = reader.read_uint(3)
    assert 0b000 <= x <= 0b111
    assert x != 0b011

    match x:
        case 0b000:
            return SampleSizeFromStreaminfo()
        case n if 0b001 <= n <= 0b111:
            return SampleSizeValue.from_bin(n)

    raise ValueError(f"Cannot read sample size: {bin(x)}")


###############################################################################

@dataclass(frozen=True)
class FrameHeader:
    blocking_strategy: BlockingStrategy
    block_size: int
    sample_rate: Optional[int]
    channels: Channels
    sample_size: Optional[int]
    coded_number: int
    crc: int


def read_frame_header(reader: Reader) -> FrameHeader:
    assert reader.read_uint(14) == 0b11111111111110
    assert reader.read_uint(1) == 0

    blocking_strategy = BlockingStrategy(reader.read_uint(1))
    _block_size = read_block_size(reader)
    _sample_rate = read_sample_rate(reader)
    channels = read_channels(reader)
    _sample_size = read_sample_size(reader)
    assert reader.read_uint(1) == 0
    coded_number = read_coded_number(reader)

    # FIXME: find a better way to make mypy happy
    block_size: int
    sample_rate: Optional[int]
    sample_size: Optional[int]

    match _block_size:
        case BlockSizeUncommon8():
            block_size = reader.read_uint(8)
        case BlockSizeUncommon16():
            block_size = reader.read_uint(16)
        case BlockSizeValue(x):
            block_size = x

    match _sample_rate:
        case SampleRateValue():
            sample_rate = _sample_rate.to_int()
        case SampleRateFromStreaminfo():
            sample_rate = None
        case SampleRateUncommon8():
            sample_rate = reader.read_uint(8)
        case SampleRateUncommon16():
            sample_rate = reader.read_uint(16)
        case SampleRateUncommon16_10():
            sample_rate = reader.read_uint(16) * 10

    match _sample_size:
        case SampleSizeFromStreaminfo():
            sample_size = None
        case SampleSizeValue():
            sample_size = _sample_size.to_int()

    crc = reader.read_uint(8)

    return FrameHeader(
        blocking_strategy,
        block_size,
        sample_rate,
        channels,
        sample_size,
        coded_number,
        crc
    )


def read_coded_number(reader: Reader):
    b0 = reader.read_uint(8)
    r = _read_coded_number_remaining(b0)
    bs = reader.read_bytes(r)

    b0_ = extract(b0, 8, r + 2, 8)
    bs_ = reduce(_read_coded_number_reduce, bs, 0)

    return (b0_ << r * 6) | bs_


def _read_coded_number_reduce(acc: int, x: int):
    return (acc << 6) | mask(6)


def _read_coded_number_remaining(b0: int):
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


###############################################################################

@dataclass(frozen=True)
class SubframeTypeConstant:
    pass


@dataclass(frozen=True)
class SubframeTypeVerbatim:
    pass


@dataclass(frozen=True)
class SubframeTypeFixed:
    order: int


@dataclass(frozen=True)
class SubframeTypeLPC:
    order: int


SubframeType = (
    SubframeTypeConstant
    | SubframeTypeVerbatim
    | SubframeTypeFixed
    | SubframeTypeLPC
)


def read_subframe_type(reader: Reader) -> SubframeType:
    x = reader.read_uint(6)
    assert (0 <= x <= 0b000001 or
            0b001000 <= x <= 0b001100 or
            0b100000 <= x <= 0b111111)

    match x:
        case 0b000000:
            return SubframeTypeConstant()
        case 0b000001:
            return SubframeTypeVerbatim()
        case n if 0b001000 <= n <= 0b001100:
            return SubframeTypeFixed(n & mask(3))
        case n if n >= 0b100000:
            return SubframeTypeLPC((n & mask(5)) + 1)

    raise ValueError(f"Cannot read subframe type: {bin(x)}")


###############################################################################

@dataclass(frozen=True)
class SubframeHeader:
    type_: SubframeType
    wasted_bits: int


def read_subframe_header(reader: Reader) -> SubframeHeader:
    assert reader.read_uint(1) == 0

    type_ = read_subframe_type(reader)
    wasted_bits = read_wasted_bits(reader)

    return SubframeHeader(type_, wasted_bits)


def read_wasted_bits(reader: Reader):
    b = reader.read_uint(1)

    if b == 0:
        return 0
    else:
        count = 0
        while reader.read_uint(1) == 0:
            count += 1
        return count


###############################################################################

@dataclass(frozen=True)
class SubframeConstant:
    sample: int

    def __repr__(self):
        return "SubframeConstant()"


@dataclass(frozen=True)
class SubframeVerbatim:
    samples: list[int]

    def __repr__(self):
        return "SubframeVerbatim()"


@dataclass(frozen=True)
class SubframeFixed:
    warmup: list[int]
    residual: list[int]

    def __repr__(self):
        return f"SubframeFixed(order={len(self.warmup)})"


@dataclass(frozen=True)
class SubframeLPC:
    warmup: list[int]
    precision: int
    shift: int
    coefficients: list[int]
    residual: list[int]

    def __repr__(self):
        return f"SubframeLPC(order={len(self.warmup)})"


Subframe = SubframeConstant | SubframeVerbatim | SubframeFixed | SubframeLPC


def read_subframe(
        reader: Reader,
        block_size: int,
        sample_size: int
) -> Subframe:
    header = read_subframe_header(reader)
    sample_size_ = sample_size - header.wasted_bits

    match header.type_:
        case SubframeTypeConstant():
            samples = reader.read_int(sample_size_)
            return SubframeConstant(samples)

        case SubframeTypeVerbatim():
            samples = [
                reader.read_int(sample_size_)
                for _ in range(block_size)
            ]
            return SubframeVerbatim(samples)

        case SubframeTypeFixed(order):
            warmup_samples = [
                reader.read_int(sample_size_)
                for _ in range(order)
            ]
            residual = read_residual(reader, block_size, order)
            return SubframeFixed(warmup_samples, residual)

        case SubframeTypeLPC(order):
            warmup_samples = [
                reader.read_int(sample_size_)
                for _ in range(order)
            ]

            precision = reader.read_uint(4)
            assert 0b0000 <= precision < 0b1111
            precision_ = precision + 1

            shift = reader.read_int(5)
            coefficients = [reader.read_int(precision_) for _ in range(order)]
            residual = read_residual(reader, block_size, order)

            return SubframeLPC(
                warmup_samples,
                precision_,
                shift,
                coefficients,
                residual
            )


def read_residual(reader: Reader, block_size: int, predictor_order: int):
    coding_method = reader.read_uint(2)
    assert 0b00 <= coding_method <= 0b01

    match coding_method:
        case 0b00:
            parameter_size = 4
        case 0b01:
            parameter_size = 5

    partition_order = reader.read_uint(4)
    partitions_count = 2 ** partition_order

    assert block_size % partitions_count == 0
    assert (block_size >> partition_order) > predictor_order

    partition0 = read_rice_partition(
        reader,
        parameter_size,
        (block_size >> partition_order) - predictor_order
    )

    partitions = [
        read_rice_partition(
            reader,
            parameter_size,
            block_size >> partition_order
        )
        for _ in range(partitions_count - 1)
    ]

    return [partition0, *partitions]


def read_rice_partition(
        reader: Reader,
        parameter_size: int,
        samples_count: int
):
    assert 4 <= parameter_size <= 5
    parameter = reader.read_uint(parameter_size)

    if parameter == mask(parameter_size):
        residual_size = reader.read_uint(5)
        return [reader.read_int(residual_size) for _ in range(samples_count)]
    else:
        return [
            read_rice_int(reader, parameter)
            for _ in range(samples_count)
        ]


def read_rice_int(reader: Reader, parameter):
    msb = 0
    while reader.read_uint(1) == 0:
        msb += 1

    lsb = reader.read_uint(parameter)

    x = (msb << parameter) | lsb
    return (x >> 1) ^ -(x & 1)


###############################################################################

@dataclass(frozen=True)
class Frame:
    header: FrameHeader
    subframes: list[Subframe]
    crc: int


def read_frame(reader: Reader, streaminfo_sample_size: int) -> Frame:
    header = read_frame_header(reader)
    sample_size = (header.sample_size or streaminfo_sample_size)

    subframes = [
        read_subframe(
            reader,
            header.block_size,
            sample_size + header.channels.decorrelation_bit[i]
        )
        for i in range(header.channels.count)
    ]

    if reader.is_byte_aligned is False:
        padding = reader.read_uint(reader.bits_until_byte_alignment)
        assert padding == 0

    crc = reader.read_uint(16)

    return Frame(header, subframes, crc)


###############################################################################

def decode(reader: Reader):
    consume_magic(reader)

    streaminfo_header = read_metadata_block_header(reader)
    assert streaminfo_header.type == MetadataBlockType.Streaminfo

    streaminfo = read_metadata_block_streaminfo(reader)
    print(streaminfo)

    print()

    if streaminfo_header.last is False:
        skip_metadata(reader)

    while True:
        frame = read_frame(reader, streaminfo.sample_size)
        print(frame.header)

        for subframe in frame.subframes:
            print(' ' * 4, subframe)

        print()
