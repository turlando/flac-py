from collections.abc import Iterator
from itertools import chain
from typing import Optional

import flac.coded_number as coded_number

from flac.binary import Reader, mask
from flac.common import (
    MetadataBlockHeader, MetadataBlockType, Streaminfo,
    BlockingStrategy,
    BlockSize, BlockSizeValue, BlockSizeUncommon8, BlockSizeUncommon16,
    SampleRate, SampleRateFromStreaminfo, SampleRateValue, SampleRateUncommon8,
    SampleRateUncommon16, SampleRateUncommon16_10,
    Channels,
    SampleSize, SampleSizeFromStreaminfo, SampleSizeValue,
    FrameHeader,
    SubframeType, SubframeTypeConstant, SubframeTypeVerbatim,
    SubframeTypeFixed, SubframeTypeLPC,
    SubframeHeader,
    Subframe, SubframeConstant, SubframeVerbatim, SubframeFixed, SubframeLPC,
    Frame
)


# -----------------------------------------------------------------------------

MAGIC = int.from_bytes(b'fLaC', byteorder='big')


def consume_magic(reader: Reader):
    assert reader.read_uint(4 * 8) == MAGIC


# -----------------------------------------------------------------------------

def read_metadata_block_header(reader: Reader) -> MetadataBlockHeader:
    return MetadataBlockHeader(
        last=reader.read_bool(),
        type=MetadataBlockType(reader.read_uint(7)),
        length=reader.read_uint(24)
    )


# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------

def skip_metadata(reader: Reader):
    while True:
        header = read_metadata_block_header(reader)
        reader.read_bytes(header.length)
        if header.last is True:
            break


# -----------------------------------------------------------------------------

def read_blocking_strategy(reader: Reader) -> BlockingStrategy:
    return BlockingStrategy(reader.read_uint(1))


# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------

def read_channels(reader: Reader) -> Channels:
    x = reader.read_uint(4)
    assert 0b0000 <= x <= 0b1010
    return Channels(x)


# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------

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
    r = coded_number.following_bytes(b0)
    bs = reader.read_bytes(r)

    return coded_number.decode(
        int.to_bytes(b0, 1, byteorder='big')
        + bs
    )


# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------

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
            return SubframeConstant(samples, block_size)

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

    # Flatten partitions of samples in a single list of samples.
    return chain.from_iterable([partition0, *partitions])


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


# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------

FIXED_PREDICTOR_COEFFICIENTS = (
    (),
    (1,),
    (2, -1),
    (3, -3, 1),
    (4, -6, 4, -1)
)


def decode_constant_subframe(subframe: SubframeConstant) -> list[int]:
    return [subframe.sample] * subframe.block_size


def decode_verbatim_subframe(subframe: SubframeVerbatim) -> list[int]:
    return subframe.samples


def decode_fixed_subframe(subframe: SubframeFixed) -> list[int]:
    order = len(subframe.warmup)
    return _decode_prediction(
        FIXED_PREDICTOR_COEFFICIENTS[order],
        0,
        subframe.warmup,
        subframe.residual
    )


def decode_lpc_subframe(subframe: SubframeLPC) -> list[int]:
    return _decode_prediction(
        subframe.coefficients,
        subframe.shift,
        subframe.warmup,
        subframe.residual
    )


def _decode_prediction(coefficients, shift, warmup, residual) -> list[int]:
    result = [*warmup, *residual]
    for i in range(len(coefficients), len(result)):
        result[i] += sum((result[i - 1 - j] * c)
                         for (j, c) in enumerate(coefficients)) >> shift
    return result


# -----------------------------------------------------------------------------

def decode_frame(frame: Frame) -> list[list[int]]:
    s = [decode_subframe(subframe) for subframe in frame.subframes]

    # Handle interchannel decorrelation
    match frame.header.channels:
        case Channels.L_S:
            return [
                s[0],
                [c0 - c1 for c0, c1 in zip(s[0], s[1])]
            ]
        case Channels.S_R:
            return [
                [c0 + c1 for c0, c1 in zip(s[0], s[1])],
                s[1]
            ]
        case Channels.M_S:
            right = [c0 - (c1 >> 1) for c0, c1 in zip(s[0], s[1])]
            left = [c1 + s for c1, s in zip(right, s[1])]
            return [left, right]
        case _:
            return s


def decode_subframe(subframe: Subframe) -> list[int]:
    match subframe:
        case SubframeConstant():
            return decode_constant_subframe(subframe)
        case SubframeVerbatim():
            return decode_verbatim_subframe(subframe)
        case SubframeFixed():
            return decode_fixed_subframe(subframe)
        case SubframeLPC():
            return decode_lpc_subframe(subframe)


# -----------------------------------------------------------------------------

def read_frames(
        reader: Reader,
        streaminfo: Streaminfo
) -> Iterator[Frame]:
    while True:
        try:
            yield read_frame(reader, streaminfo.sample_size)
        except EOFError:
            return
