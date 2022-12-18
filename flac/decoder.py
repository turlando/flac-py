from dataclasses import dataclass
from enum import Enum
from functools import reduce
from typing import Optional, Any

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
    depth: int  ## TODO: rename to sample_size?
    samples: int
    md5: bytes


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

    class FromStreaminfo:
        pass

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

        def to_int(self):
            match self.value:
                case self.V_88_2_kHz:
                    return 88_200
                case self.V_176_4_kHz:
                    return 176_400
                case self.V_192_kHz:
                    return 192_000
                case self.V_8_kHz:
                    return 8_000
                case self.V_16_kHz:
                    return 16_000
                case self.V_22_05_kHz:
                    return 22_050
                case self.V_24_kHz:
                    return 24_000
                case self.V_32_kHz:
                    return 32_000
                case self.V_44_1_kHz:
                    return 44_100
                case self.V_48_kHz:
                    return 48_000
                case self.V_96_kHz:
                    return 96_000

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

    @property
    def count(self):
        match self:
            case self.M:
                return 1
            case self.L_R:
                return 2
            case self.L_R_C:
                return 3
            case self.FL_FR_BL_BR:
                return 4
            case self.FL_FR_FC_BL_BR:
                return 5
            case self.FL_FR_FC_LFE_BL_BR:
                return 6
            case self.FL_FR_FC_LFE_BC_SL_SR:
                return 7
            case self.FL_FR_FC_LFE_BL_BR_SL_SR:
                return 8
            case self.L_S:
                return 2
            case self.S_R:
                return 2
            case self.M_S:
                return 2


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

        def to_int(self):
            match self.value:
                case self.V_8:
                    return 8
                case self.V_12:
                    return 12
                case self.V_16:
                    return 16
                case self.V_20:
                    return 20
                case self.V_24:
                    return 24
                case self.V_32:
                    return 32


@dataclass
class FrameHeader:
    blocking_strategy: BlockingStrategy
    block_size: int
    sample_rate: Optional[int]
    channels: Channels
    sample_size: Optional[int]
    coded_number: int
    crc: int


class SubframeType:
    def __new__(cls, x: int):
        assert (0 <= x <= 0b000001 or
                0b001000 <= x <= 0b001100 or
                0b100000 <= x <= 0b111111)

        match x:
            case 0b000000:
                return cls.Constant()
            case 0b000001:
                return cls.Verbatim()
            case n if 0b001000 <= n <= 0b001100:
                return cls.Fixed(n & mask(3))
            case n if n >= 0b100000:
                return cls.LPC((n & mask(5)) + 1)

    class Constant:
        pass

    class Verbatim:
        pass

    @dataclass
    class Fixed:
        order: int

    @dataclass
    class LPC:
        order: int


@dataclass
class SubframeHeader:
    type_: SubframeType
    wasted_bits: int


class Subframe:
    @dataclass
    class Constant:
        sample: int

    @dataclass
    class Verbatim:
        samples: list[int]

    @dataclass
    class Fixed:
        warmup: list[int]
        resitual: list[int]

    @dataclass
    class LPC:
        warmup: list[int]
        precision: int
        shift: int
        coefficients: list[int]
        residual: list[int]


@dataclass
class Frame:
    header: FrameHeader
    subframes: Any #: list[Subframe]


###############################################################################

def decode_frame(reader: Reader, streaminfo_sample_size: int):
    header = decode_frame_header(reader)

    sample_size = header.sample_size or streaminfo_sample_size

    subframes = [
        decode_subframe(reader, header.block_size, sample_size)
        for _ in range(header.channels.count)
    ]

    return Frame(header, subframes)


def decode_frame_header(reader: Reader):
    assert reader.read(14) == 0b11111111111110
    assert reader.read(1) == 0b0

    blocking_strategy = BlockingStrategy(reader.read(1))
    _block_size = BlockSize(reader.read(4))
    _sample_rate = SampleRate(reader.read(4))
    channels = Channels(reader.read(4))
    _sample_size = SampleSize(reader.read(3))
    assert reader.read(1) == 0b0
    coded_number = decode_coded_number(reader)

    # FIXME: find a better way to make mypy happy
    block_size: int
    sample_rate: Optional[int]
    sample_size: Optional[int]

    match _block_size:
        case BlockSize.Uncommon8():
            block_size = reader.read(8)
        case BlockSize.Uncommon16():
            block_size = reader.read(16)
        case BlockSize.Value(x):
            block_size = x

    match _sample_rate:
        case SampleRate.Value():
            sample_rate = _sample_rate.to_int()
        case SampleRate.FromStreaminfo:
            sample_rate = None
        case SampleRate.Uncommon8():
            sample_rate = reader.read(8)
        case SampleRate.Uncommon16():
            sample_rate = reader.read(16)
        case SampleRate.Uncommon16_10():
            sample_rate = reader.read(16) * 10

    match _sample_size:
        case SampleSize.FromStreaminfo():
            sample_size = None
        case SampleSize.Value():
            sample_size = _sample_size.to_int()

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


def decode_subframe(reader: Reader, block_size: int, sample_size: int):
    header = decode_subframe_header(reader)
    sample_size_ = sample_size - header.wasted_bits

    match header.type_:
        case SubframeType.Constant():
            samples = reader.read_int(sample_size_)
            return Subframe.Constant(samples)

        case SubframeType.Verbatim():
            samples = [reader.read_int(sample_size) for _ in range(block_size)]
            return Subframe.Verbatim(samples)

        case SubframeType.Fixed(order):
            warmup_samples = [
                reader.read_int(sample_size)
                for _ in range(order)
            ]
            residual = decode_residual(reader, sample_size, order)
            return Subframe.Fixed(warmup_samples, residual)

        case SubframeType.LPC(order):
            warmup_samples = [
                reader.read_int(sample_size)
                for _ in range(order)
            ]

            precision = reader.read(4)
            assert 0b0000 <= precision < 0b1111
            shift = reader.read_int(5)
            coefficients = [reader.read_int(precision) for _ in range(order)]
            residual = decode_residual(reader, sample_size, order)
            return Subframe.LPC(
                warmup_samples,
                precision,
                shift,
                coefficients,
                residual
            )


def decode_subframe_header(reader: Reader):
    assert reader.read(1) == 0

    type_ = SubframeType(reader.read(6))
    wasted_bits = decode_wasted_bits(reader)

    return SubframeHeader(type_, wasted_bits)


def decode_wasted_bits(reader: Reader):
    b = reader.read_bool()

    if b is False:
        return 0
    else:
        count = 0
        while reader.read(1) == 0:
            count += 1
        return count


def decode_residual(reader: Reader, block_size: int, predictor_order: int):
    coding_method = reader.read(2)
    assert 0b00 <= coding_method <= 0b01

    match coding_method:
        case 0b00:
            parameter_size = 4
        case 0b01:
            parameter_size = 5

    partition_order = reader.read(4)

    partition0 = decode_rice_partition(
        reader,
        parameter_size,
        (block_size >> partition_order) - predictor_order
    )

    partitions = [
        decode_rice_partition(
            reader,
            parameter_size,
            block_size >> partition_order
        )
        for _ in range((2 ** partition_order) - 1)
    ]

    return [partition0, *partitions]


def decode_rice_partition(
        reader: Reader,
        parameter_size: int,
        samples_count: int
):
    assert 4 <= parameter_size <= 5
    parameter = reader.read(parameter_size)

    if parameter == mask(parameter_size):
        residual_size = reader.read(5)
        return [reader.read_int(residual_size) for _ in range(samples_count)]
    else:
        return [
            decode_rice_int(reader, parameter)
            for _ in range(samples_count)
        ]


def decode_rice_int(reader: Reader, parameter):
    msb = 0
    while reader.read(1) == 0:
        msb += 1

    lsb = reader.read(parameter)

    x = (msb << parameter) | lsb
    return (x >> 1) ^ -(x & 1)


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

    frame = decode_frame(reader, streaminfo.depth)
    print(frame)
