from dataclasses import dataclass
from typing import Iterator, Optional

import flac.coded_number as coded_number

from flac.binary import Put
from flac.crc import crc8, crc16
from flac.utils import batch, group

from flac.common import (
    MAGIC, FRAME_SYNC_CODE,
    CRC8_POLYNOMIAL, CRC16_POLYNOMIAL,
    BLOCK_SIZE_ENCODING, SAMPLE_RATE_VALUE_ENCODING, SAMPLE_SIZE_ENCODING,
    CHANNELS_ENCODING,
    MetadataBlockHeader, MetadataBlockType, Streaminfo,
    FrameHeader, SubframeHeader,
    SubframeType, SubframeTypeConstant, SubframeTypeVerbatim,
    SubframeTypeFixed, SubframeTypeLPC,
    SubframeVerbatim,
    BlockingStrategy, Channels,
    BlockSize, BlockSizeValue, BlockSizeUncommon8, BlockSizeUncommon16,
    SampleRate, SampleRateFromStreaminfo, SampleRateValue, SampleRateUncommon8,
    SampleRateUncommon16, SampleRateUncommon16_10,
    SampleSize, SampleSizeFromStreaminfo, SampleSizeValue
)


# -----------------------------------------------------------------------------

@dataclass
class EncoderParameters:
    block_size: int


# -----------------------------------------------------------------------------

def encode(
        sample_rate: int,
        sample_size: int,
        channels: int,
        frames: int,
        samples: Iterator[list[int]],
        parameters: EncoderParameters
) -> Iterator[bytes]:
    yield MAGIC

    # -------------------------------------------------------------------------

    yield put_metadata_block_header(
        MetadataBlockHeader(
            last=True,
            type=MetadataBlockType.Streaminfo,
            length=34
        )
    ).buffer

    yield put_metadata_block_streaminfo(
        Streaminfo(
            min_block_size=parameters.block_size,
            max_block_size=parameters.block_size,
            min_frame_size=0,
            max_frame_size=0,
            sample_rate=sample_rate,
            channels=channels,
            sample_size=sample_size,
            samples=frames,
            md5=bytes(16)
        )
    ).buffer

    # -------------------------------------------------------------------------

    for i, xs in enumerate(batch(samples, parameters.block_size)):
        block_size_ = len(xs)

        frame_put = put_frame_header(
            FrameHeader(
                blocking_strategy=BlockingStrategy.Fixed,
                block_size=block_size_,
                sample_rate=None,
                channels=Channels.L_R,
                sample_size=None,
                coded_number=i,
            )
        )

        for c in range(channels):
            _put_subframe_header(
                frame_put,
                SubframeHeader(
                    type_=SubframeTypeVerbatim(),
                    wasted_bits=0
                )
            )

            _put_subframe_verbatim(
                frame_put,
                SubframeVerbatim(
                    samples=[x[c] for x in xs]
                ),
                sample_size
            )

        frame_put.uint(crc16(frame_put.buffer, CRC16_POLYNOMIAL), 16)

        yield frame_put.buffer


# -----------------------------------------------------------------------------

def put_metadata_block_header(header: MetadataBlockHeader) -> Put:
    put = Put()
    put.bool(header.last)
    put.uint(header.type.value, 7)
    put.uint(header.length, 24)
    return put


def put_metadata_block_streaminfo(streaminfo: Streaminfo) -> Put:
    put = Put()
    put.uint(streaminfo.min_block_size, 16)
    put.uint(streaminfo.max_block_size, 16)
    put.uint(streaminfo.min_frame_size, 24)
    put.uint(streaminfo.max_frame_size, 24)
    put.uint(streaminfo.sample_rate, 20)
    put.uint(streaminfo.channels - 1, 3)
    put.uint(streaminfo.sample_size - 1, 5)
    put.uint(streaminfo.samples, 36)
    put.bytes(streaminfo.md5)  # 16 bytes
    return put


# -----------------------------------------------------------------------------

def put_frame_header(header: FrameHeader) -> Put:
    put = Put()

    _put_frame_sync_code(put)
    _put_blocking_strategy(put, header.blocking_strategy)

    _block_size = encode_block_size(header.block_size)
    _sample_rate = encode_sample_rate(header.sample_rate)
    _sample_size = encode_sample_size(header.sample_size)

    _put_block_size(put, _block_size)
    _put_sample_rate(put, _sample_rate)
    _put_channels(put, header.channels)
    _put_sample_size(put, _sample_size)

    put.uint(0b0, 1)

    _put_coded_number(put, header.coded_number)

    match _block_size:
        case BlockSizeUncommon8():
            assert header.block_size is not None
            put.uint(header.block_size - 1, 8)
        case BlockSizeUncommon16():
            assert header.block_size is not None
            put.uint(header.block_size - 1, 16)

    match _sample_rate:
        case SampleRateUncommon8():
            assert header.sample_rate is not None
            put.uint(header.sample_rate, 8)
        case SampleRateUncommon16():
            assert header.sample_rate is not None
            put.uint(header.sample_rate, 16)
        case SampleRateUncommon16_10():
            assert header.sample_rate is not None
            put.uint(header.sample_rate // 10, 16)

    put.uint(crc8(put.buffer, CRC8_POLYNOMIAL), 8)

    return put


def _put_frame_sync_code(put: Put):
    put.uint(FRAME_SYNC_CODE, 15)


def _put_blocking_strategy(put: Put, blocking_strategy: BlockingStrategy):
    put.uint(blocking_strategy.value, 1)


def encode_block_size(size: int) -> BlockSize:
    if (maybe_size := BLOCK_SIZE_ENCODING.get(size)) is not None:
        return BlockSizeValue(maybe_size)

    match size.bit_length():
        case n if 0 < n <= 8:
            return BlockSizeUncommon8()
        case n if 8 < n <= 16:
            return BlockSizeUncommon16()

    raise ValueError(f"Cannot encode block size: {size}")


def encode_sample_rate(sample_rate: Optional[int]) -> SampleRate:
    match sample_rate:
        case None:
            return SampleRateFromStreaminfo()
        case n if n in SampleRateValue.values():
            return SampleRateValue(n)
        case n if 0 < n <= 8:
            return SampleRateUncommon8()
        case n if n % 10 == 0 and 8 < n <= 16:
            return SampleRateUncommon16_10()
        case n if 8 < n <= 16:
            return SampleRateUncommon16()

    raise ValueError(f"Cannot encode sample rate: {sample_rate}")


def encode_sample_size(size: Optional[int]) -> SampleSize:
    if size is None:
        return SampleSizeFromStreaminfo()
    return SampleSizeValue(size)


def _put_block_size(put: Put, size: BlockSize):
    match size:
        case BlockSizeValue(x):
            put.uint(x, 4)
        case BlockSizeUncommon8():
            put.uint(0b0110, 4)
        case BlockSizeUncommon16():
            put.uint(0b0111, 4)


def _put_sample_rate(put: Put, sample_rate: SampleRate):
    match sample_rate:
        case SampleRateFromStreaminfo():
            put.uint(0b0000, 4)
        case SampleRateValue() as v:
            put.uint(SAMPLE_RATE_VALUE_ENCODING[v], 4)
        case SampleRateUncommon8():
            put.uint(0b1100, 4)
        case SampleRateUncommon16():
            put.uint(0b1101, 4)
        case SampleRateUncommon16_10():
            put.uint(0b1110, 4)


def _put_channels(put: Put, channels: Channels):
    put.uint(CHANNELS_ENCODING[channels], 4)


def _put_sample_size(put: Put, sample_size: SampleSize):
    match sample_size:
        case SampleSizeFromStreaminfo():
            put.uint(0b000, 3)
        case SampleSizeValue() as v:
            put.uint(SAMPLE_SIZE_ENCODING[v], 3)


def _put_coded_number(put: Put, x: int):
    x_ = coded_number.encode(x)
    put.bytes(x_)


# -----------------------------------------------------------------------------

def _put_subframe_header(put: Put, header: SubframeHeader):
    put.uint(0b0, 1)
    _put_subframe_type(put, header.type_)
    put.uint(0b0, 1)  # zero wasted bits


def _put_subframe_type(put: Put, type: SubframeType):
    match type:
        case SubframeTypeConstant():
            put.uint(0b000000, 6)
        case SubframeTypeVerbatim():
            put.uint(0b000001, 6)
        case SubframeTypeFixed(order):
            assert order <= 4
            put.uint(0b001000 | order, 6)
        case SubframeTypeLPC(order):
            put.uint(0b100000 | order - 1, 6)


def _put_subframe_verbatim(
        put: Put,
        subframe: SubframeVerbatim,
        sample_size: int  # bits
):
    for s in subframe.samples:
        put.uint(s, sample_size)


# -----------------------------------------------------------------------------

def find_rice_partition_order(
        residuals: list[int],
        block_size: int,
        predictor_order: int,
        partition_order_range: range,
        rice_parameter_range: range
) -> tuple[int, list[int]]:
    # The partition order MUST be so that the block size is evenly divisible by
    # the number of partitions. The partition order also MUST be so that the
    # (block size >> partition order) is larger than the predictor order.
    candidate_orders = [o for o in partition_order_range
                        if (block_size % (1 << o) == 0
                            and (block_size >> o) > predictor_order)]
    assert len(candidate_orders) > 0

    def partitions_size(order: int) -> tuple[int, list[int]]:
        partition0_samples_count = (block_size >> order) - predictor_order
        partitions_samples_count = block_size >> order

        partition0_samples = residuals[:partition0_samples_count]
        partitions_samples = group(residuals[partition0_samples_count:],
                                   partitions_samples_count)

        partition0_size = rice_partition_size(partition0_samples,
                                              rice_parameter_range)
        partitions_size = (rice_partition_size(samples, rice_parameter_range)
                           for samples in partitions_samples)

        # Second element in the tuple is the size
        total_size = partition0_size[1] + sum(x[1] for x in partitions_size)

        # First element in the tuple is a list of the rice parameters for
        # each partition
        partitions_orders = [partition0_size[0],
                             *[x[0] for x in partitions_size]]

        return (total_size, partitions_orders)

    sizes = ((o, *partitions_size(o)) for o in candidate_orders)

    # (partition order, total size of the partitions in bits,
    # list of rice parameters for each partition)
    order = min(sizes, key=lambda x: x[1])

    # (partition order, list of the rice parameters for each partition)
    return (order[0], order[2])


def rice_partition_size(
        residuals: list[int],
        rice_parameter_range: range
) -> tuple[int, int]:
    """
    Return (best rice parameter for given residuals, total size of the
    partition in bits)
    """
    (parameter, size) = find_rice_parameter(residuals, rice_parameter_range)
    parameter_size = 5 if parameter > 14 else 4

    partition_size = (
        4  # number of bits to encode the partition order
        + parameter_size  # number of bits to encode the rice parameter
        + size
    )

    return (parameter, partition_size)


def find_rice_parameter(
        residuals: list[int],
        rice_parameter_range: range
) -> tuple[int, int]:
    """
    Return (best rice parameter for given residuals, total size of rice
    encoded residuals in bits)
    """
    sizes = [sum(rice_size(x, parameter) for x in residuals)
             for parameter in rice_parameter_range]
    parameter = min(rice_parameter_range,
                    key=lambda x: sizes[rice_parameter_range.index(x)])
    return (parameter, sizes[rice_parameter_range.index(parameter)])


def rice_size(x: int, parameter: int) -> int:
    "Size of rice-encoded number in bits"
    m = 1 << parameter
    q = x // m
    return q + 1 + parameter
