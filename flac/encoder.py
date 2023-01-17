from dataclasses import dataclass
from math import floor, log2
from typing import Iterator, Optional

import flac.coded_number as coded_number

from flac.binary import Put
from flac.crc import crc8, crc16
from flac.utils import batch, group, zigzag_encode

from flac.common import (
    MAGIC, FRAME_SYNC_CODE,
    CRC8_POLYNOMIAL, CRC16_POLYNOMIAL,
    FIXED_PREDICTOR_COEFFICIENTS,
    BLOCK_SIZE_ENCODING, SAMPLE_RATE_VALUE_ENCODING, SAMPLE_SIZE_ENCODING,
    CHANNELS_ENCODING,
    MetadataBlockHeader, MetadataBlockType, Streaminfo,
    FrameHeader, SubframeHeader,
    SubframeType, SubframeTypeConstant, SubframeTypeVerbatim,
    SubframeTypeFixed, SubframeTypeLPC,
    SubframeVerbatim, SubframeFixed,
    BlockingStrategy, Channels,
    BlockSize, BlockSizeValue, BlockSizeUncommon8, BlockSizeUncommon16,
    SampleRate, SampleRateFromStreaminfo, SampleRateValue, SampleRateUncommon8,
    SampleRateUncommon16, SampleRateUncommon16_10,
    SampleSize, SampleSizeFromStreaminfo, SampleSizeValue,
    Residual, RiceCodingMethod, RicePartition, EscapedPartition
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
            # header, subframe = encode_subframe_verbatim([x[c] for x in xs])
            # _put_subframe_verbatim(frame_put, subframe, sample_size)

            header, subframe = encode_subframe_fixed([x[c] for x in xs], 4)
            _put_subframe_header(frame_put, header)
            _put_subframe_fixed(
                frame_put,
                subframe,
                block_size_, sample_size,
                range(0, 17), range(0, 15)
            )

        # Frame padding
        frame_put.uint(0b0, frame_put.bits_until_alignment)

        # Frame footer
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

def encode_subframe_verbatim(
        samples: list[int]
) -> tuple[SubframeHeader, SubframeVerbatim]:
    header = SubframeHeader(type_=SubframeTypeVerbatim(), wasted_bits=0)
    subframe = SubframeVerbatim(samples)
    return (header, subframe)


def encode_subframe_fixed(
        samples: list[int],
        order: int
) -> tuple[SubframeHeader, SubframeFixed]:
    assert 0 <= order <= 4

    coefficients = FIXED_PREDICTOR_COEFFICIENTS[order]
    warmup = samples[:order]
    residual = prediction_residual(samples, coefficients)

    header = SubframeHeader(SubframeTypeFixed(order=order), 0)
    subframe = SubframeFixed(warmup, residual)

    return (header, subframe)


def prediction_residual(samples, coefficients):
    order = len(coefficients)
    return [
        (samples[i]
         - sum(samples[i - 1 - j] * c for j, c in enumerate(coefficients)))
        for i in range(order, len(samples))
    ]


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
            assert 0 <= order <= 4
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


def _put_subframe_fixed(
        put: Put,
        subframe: SubframeFixed,
        block_size: int,
        sample_size: int,
        partition_order_range: range,
        rice_parameter_range: range
):
    residual = encode_residual(
        subframe.residual,
        sample_size
    )

    for s in subframe.warmup:
        put.uint(s, sample_size)
    put_residual(put, residual, sample_size)


# -----------------------------------------------------------------------------

def encode_residual(
        samples: list[int],
        sample_size: int
) -> Residual:
    samples_ = [zigzag_encode(x, sample_size) for x in samples]
    rice_parameter = find_rice_parameter(samples_)

    return Residual(
        coding_method=RiceCodingMethod.Rice4Bit,
        partitions=[RicePartition(rice_parameter, samples_)]
    )


def find_rice_parameter(
        samples: list[int],
) -> int:
    # Code stolen from libFLAC, stream_encoder.c, set_partitioned_rice.
    # The following comment is present, introducing some code that has been
    # disabled and some more code that I have not spent enough time to
    # fully understand. What I'm doing is blindly implementing what the comment
    # states. Here we go.
    #
    # we are basically calculating the size in bits of the
    # average residual magnitude in the partition:
    #   rice_parameter = floor(log2(mean/partition_samples))
    # 'mean' is not a good name for the variable, it is
    # actually the sum of magnitudes of all residual values
    # in the partition, so the actual mean is
    # mean/partition_samples
    #
    # I take some freedom on such notes, possibly being completely wrong as I
    # yet have to master this aspect: I don't abs the samples, assuming zig-zag
    # encoding has already been performed.
    return floor(log2(sum(samples)/len(samples)))


# -----------------------------------------------------------------------------

def put_residual(put: Put, residual: Residual, sample_size: int):
    put_residual_coding_metod(put, residual.coding_method)
    put.uint(residual.partition_order, 4)

    for partition in residual.partitions:
        match partition:
            case EscapedPartition():
                raise NotImplementedError()
            case RicePartition() as p:
                put_rice_partition(put, p, residual.coding_method, sample_size)


def put_residual_coding_metod(put: Put, coding_method: RiceCodingMethod):
    match coding_method:
        case RiceCodingMethod.Rice4Bit:
            put.uint(0b00, 2)
        case RiceCodingMethod.Rice5Bit:
            put.uint(0b01, 2)


def put_rice_partition(
        put: Put,
        partition: RicePartition,
        coding_method: RiceCodingMethod,
        sample_size: int
):
    put.uint(partition.parameter, coding_method.value)

    for x in partition.residual:
        assert x >= 0
        put_rice_int(put, x, partition.parameter)


def put_rice_int(put: Put, x: int, parameter: int):
    m = 1 << parameter
    q = x // m

    put.uint(0b0, q)
    put.uint(0b1, 1)

    for i in reversed(range(parameter)):
        put.uint((x >> i) & 1, 1)
