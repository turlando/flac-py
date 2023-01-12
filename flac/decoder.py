from itertools import chain
from io import BufferedReader
from typing import Iterator, Optional

import flac.coded_number as coded_number

from flac.binary import Get, mask

from flac.common import (
    MAGIC, FRAME_SYNC_CODE, FIXED_PREDICTOR_COEFFICIENTS,
    SAMPLE_RATE_VALUE_DECODING, SAMPLE_SIZE_DECODING,
    CHANNELS_DECODING,
    MetadataBlockHeader, MetadataBlockType, Streaminfo,
    Frame, FrameHeader, Subframe, SubframeHeader,
    SubframeType, SubframeTypeConstant, SubframeTypeVerbatim,
    SubframeTypeFixed, SubframeTypeLPC,
    SubframeConstant, SubframeVerbatim, SubframeFixed, SubframeLPC,
    BlockingStrategy, Channels,
    BlockSize, BlockSizeValue, BlockSizeUncommon8, BlockSizeUncommon16,
    SampleRate, SampleRateFromStreaminfo, SampleRateValue, SampleRateUncommon8,
    SampleRateUncommon16, SampleRateUncommon16_10,
    SampleSize, SampleSizeFromStreaminfo, SampleSizeValue
)


# -----------------------------------------------------------------------------

def decode(
        buffer: BufferedReader
) -> tuple[int, int, int, int, Iterator[list[int]]]:
    get = Get(buffer)

    assert get.bytes(4) == MAGIC

    streaminfo_header = get_metadata_block_header(get)
    assert streaminfo_header.type == MetadataBlockType.Streaminfo

    streaminfo = get_metadata_block_streaminfo(get)

    if streaminfo_header.last is False:
        skip_metadata(get)

    def samples() -> Iterator[list[int]]:
        for frame in get_frames(get, streaminfo):
            samples = decode_frame(frame)

            # from [[l1, l2, l3, ...], [r1, r2, r3, ...]]
            # to [[l1, r1], [l2, r2], [l3, r3], ...]
            samples_ = zip(*samples)

            for s in samples_:
                yield list(s)

    return (
        streaminfo.sample_rate,
        streaminfo.sample_size,
        streaminfo.channels,
        streaminfo.samples,
        samples()
    )


# -----------------------------------------------------------------------------

def get_metadata_block_header(get: Get) -> MetadataBlockHeader:
    return MetadataBlockHeader(
        last=get.bool(),
        type=MetadataBlockType(get.uint(7)),
        length=get.uint(24)
    )


def get_metadata_block_streaminfo(get: Get) -> Streaminfo:
    return Streaminfo(
        min_block_size=get.uint(16),
        max_block_size=get.uint(16),
        min_frame_size=get.uint(24),
        max_frame_size=get.uint(24),
        sample_rate=get.uint(20),
        channels=get.uint(3) + 1,
        sample_size=get.uint(5) + 1,
        samples=get.uint(36),
        md5=get.bytes(16)
    )


def skip_metadata(get: Get):
    while True:
        header = get_metadata_block_header(get)
        get.bytes(header.length)
        if header.last is True:
            break


# -----------------------------------------------------------------------------

def get_frames(
        get: Get,
        streaminfo: Streaminfo
) -> Iterator[Frame]:
    while True:
        try:
            yield get_frame(get, streaminfo.sample_size)
        except EOFError:
            return


def get_frame(get: Get, streaminfo_sample_size: int) -> Frame:
    header = get_frame_header(get)
    sample_size = (header.sample_size or streaminfo_sample_size)

    subframes = [
        get_subframe(
            get,
            header.block_size,
            sample_size + header.channels.decorrelation_bit[i]
        )
        for i in range(header.channels.count)
    ]

    if get.is_aligned is False:
        padding = get.uint(get.bits_until_alignment)
        assert padding == 0

    crc = get.uint(16)

    return Frame(header, subframes, crc)


def get_frame_header(get: Get) -> FrameHeader:
    assert get.uint(15) == FRAME_SYNC_CODE

    blocking_strategy = get_blocking_strategy(get)
    _block_size = get_block_size(get)
    _sample_rate = get_sample_rate(get)
    channels = get_channels(get)
    _sample_size = get_sample_size(get)
    assert get.uint(1) == 0
    coded_number = get_coded_number(get)

    # FIXME: find a better way to make mypy happy
    block_size: int
    sample_rate: Optional[int]
    sample_size: Optional[int]

    match _block_size:
        case BlockSizeUncommon8():
            block_size = get.uint(8)
        case BlockSizeUncommon16():
            block_size = get.uint(16)
        case BlockSizeValue(x):
            block_size = x

    match _sample_rate:
        case SampleRateValue():
            sample_rate = _sample_rate.value
        case SampleRateFromStreaminfo():
            sample_rate = None
        case SampleRateUncommon8():
            sample_rate = get.uint(8)
        case SampleRateUncommon16():
            sample_rate = get.uint(16)
        case SampleRateUncommon16_10():
            sample_rate = get.uint(16) * 10

    match _sample_size:
        case SampleSizeFromStreaminfo():
            sample_size = None
        case SampleSizeValue():
            sample_size = _sample_size.value

    crc = get.uint(8)

    return FrameHeader(
        blocking_strategy,
        block_size,
        sample_rate,
        channels,
        sample_size,
        coded_number,
        crc
    )


def get_blocking_strategy(get: Get) -> BlockingStrategy:
    return BlockingStrategy(get.uint(1))


def get_block_size(get: Get) -> BlockSize:
    x = get.uint(4)
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


def get_sample_rate(get: Get) -> SampleRate:
    x = get.uint(4)
    assert 0b0000 <= x < 0b1111

    match x:
        case 0b0000:
            return SampleRateFromStreaminfo()
        case n if 0b0001 <= n <= 0b1011:
            return SAMPLE_RATE_VALUE_DECODING[n]
        case 0b1100:
            return SampleRateUncommon8()
        case 0b1101:
            return SampleRateUncommon16()
        case 0b1110:
            return SampleRateUncommon16_10()

    raise ValueError(f"Cannot read sample rate: {bin(x)}")


def get_channels(get: Get) -> Channels:
    x = get.uint(4)
    assert 0b0000 <= x <= 0b1010
    return CHANNELS_DECODING[x]


def get_sample_size(get: Get) -> SampleSize:
    x = get.uint(3)
    assert 0b000 <= x <= 0b111
    assert x != 0b011

    match x:
        case 0b000:
            return SampleSizeFromStreaminfo()
        case n if 0b001 <= n <= 0b111:
            return SAMPLE_SIZE_DECODING[n]

    raise ValueError(f"Cannot read sample size: {bin(x)}")


def get_coded_number(get: Get):
    b0 = get.uint(8)
    b0_ = int.to_bytes(b0, 1, byteorder='big')

    r = coded_number.following_bytes(b0)

    if r > 0:
        bs = get.bytes(r)
        return coded_number.decode(
            b0_ + bs
        )
    else:
        return coded_number.decode(b0_)


# -----------------------------------------------------------------------------

def get_subframe(
        get: Get,
        block_size: int,
        sample_size: int
) -> Subframe:
    header = get_subframe_header(get)
    sample_size_ = sample_size - header.wasted_bits

    match header.type_:
        case SubframeTypeConstant():
            samples = get.sint(sample_size_)
            return SubframeConstant(samples, block_size)

        case SubframeTypeVerbatim():
            samples = [
                get.sint(sample_size_)
                for _ in range(block_size)
            ]
            return SubframeVerbatim(samples)

        case SubframeTypeFixed(order):
            warmup_samples = [
                get.sint(sample_size_)
                for _ in range(order)
            ]
            residual = get_residual(get, block_size, order)
            return SubframeFixed(warmup_samples, residual)

        case SubframeTypeLPC(order):
            warmup_samples = [
                get.sint(sample_size_)
                for _ in range(order)
            ]

            precision = get.uint(4)
            assert 0b0000 <= precision < 0b1111
            precision_ = precision + 1

            shift = get.sint(5)
            coefficients = [get.sint(precision_) for _ in range(order)]
            residual = get_residual(get, block_size, order)

            return SubframeLPC(
                warmup_samples,
                precision_,
                shift,
                coefficients,
                residual
            )


def get_subframe_header(get: Get) -> SubframeHeader:
    assert get.uint(1) == 0

    type_ = get_subframe_type(get)
    wasted_bits = get_wasted_bits(get)

    return SubframeHeader(type_, wasted_bits)


def get_subframe_type(get: Get) -> SubframeType:
    x = get.uint(6)
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


def get_wasted_bits(get: Get):
    b = get.uint(1)

    if b == 0:
        return 0
    else:
        count = 0
        while get.uint(1) == 0:
            count += 1
        return count


def get_residual(get: Get, block_size: int, predictor_order: int) -> list[int]:
    coding_method = get.uint(2)
    assert 0b00 <= coding_method <= 0b01

    match coding_method:
        case 0b00:
            parameter_size = 4
        case 0b01:
            parameter_size = 5

    partition_order = get.uint(4)
    partitions_count = 2 ** partition_order

    assert block_size % partitions_count == 0
    assert (block_size >> partition_order) > predictor_order

    partition0 = get_rice_partition(
        get,
        parameter_size,
        (block_size >> partition_order) - predictor_order
    )

    partitions = [
        get_rice_partition(
            get,
            parameter_size,
            block_size >> partition_order
        )
        for _ in range(partitions_count - 1)
    ]

    # Flatten partitions of samples in a single list of samples.
    return list(chain.from_iterable([partition0, *partitions]))


def get_rice_partition(
        get: Get,
        parameter_size: int,
        samples_count: int
):
    assert 4 <= parameter_size <= 5
    parameter = get.uint(parameter_size)

    if parameter == mask(parameter_size):
        residual_size = get.uint(5)
        return [get.sint(residual_size) for _ in range(samples_count)]
    else:
        return [
            get_rice_int(get, parameter)
            for _ in range(samples_count)
        ]


def get_rice_int(get: Get, parameter):
    msb = 0
    while get.uint(1) == 0:
        msb += 1

    lsb = get.uint(parameter)

    x = (msb << parameter) | lsb
    return (x >> 1) ^ -(x & 1)


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
