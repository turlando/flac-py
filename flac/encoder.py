from dataclasses import dataclass
from math import cos, floor, log2, pi
from typing import Iterator, Optional, Sequence

import flac.coded_number as coded_number

from flac.binary import Put
from flac.crc import crc8, crc16
from flac.utils import batch, clamp, group, zigzag_encode

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
    SubframeVerbatim, SubframeFixed, SubframeLPC,
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
    rice_partition_order: range
    lpc_order: range
    qlp_precision: int

    def __post_init__(self):
        assert self.lpc_order.start == 0
        assert self.lpc_order.stop <= 33  # Max LPC order is 32
        assert self.qlp_precision >= 5


# -----------------------------------------------------------------------------

def encode(
        sample_rate: int,
        sample_size: int,
        channels: int,
        frames: int,
        samples: Iterator[list[int]],
        parameters: EncoderParameters
) -> Iterator[bytes]:
    if sample_rate <= 48_000:
        assert parameters.lpc_order.stop <= 13

    # -------------------------------------------------------------------------

    yield MAGIC

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
            samples_ = [x[c] for x in xs]

            # 1. Build fixed and lpc subframes.
            #
            # For now we're ignoring verbatim subframes, assuming a compression
            # will always yield better results.
            # Might not be true under some pathological conditions, but a
            # reliable way of determining the size of each subframe has not
            # been implemented yet.
            # Actually, since we're at it...
            # TODO: Implement a reliable way of determining the size of
            # subframe structures. This could be easily done by counting the
            # written bytes in binary.Put and using a different Put instance
            # for each subframe+header. Once the smallest has been found there
            # should be a way of "glueing" such Put instance into frame_put.
            # For now, we're just measuring the magnitude of the residual in
            # order to find the best compression method.
            #
            # verbatim_header, verbatim_subframe = encode_subframe_verbatim(
            #     samples_
            # )
            #
            # _put_subframe_header(frame_put, verbatim_header)
            # _put_subframe_verbatim(frame_put, verbatim_subframe, sample_size)

            fixed_header, fixed_subframe = encode_subframe_fixed(samples_)

            lpc_header, lpc_subframe = encode_subframe_lpc(
                samples_, parameters.lpc_order, parameters.qlp_precision
            )

            # 2. Pick the smallest subframe.

            fixed_size = sum(abs(x) for x in fixed_subframe.residual)
            lpc_size = sum(abs(x) for x in lpc_subframe.residual)

            if fixed_size < lpc_size:
                _put_subframe_header(frame_put, fixed_header)
                _put_subframe_fixed(
                    frame_put,
                    fixed_subframe,
                    block_size_,
                    sample_size,
                    parameters.rice_partition_order
                )
            elif lpc_size < fixed_size:
                _put_subframe_header(frame_put, lpc_header)
                _put_subframe_lpc(
                    frame_put,
                    lpc_subframe,
                    block_size_,
                    sample_size,
                    parameters.rice_partition_order
                )
            else:
                assert False

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
) -> tuple[SubframeHeader, SubframeFixed]:
    if len(samples) <= 4:
        # If the block size is <= 4 then use order zero
        order = 0
        warmup = []
        residual = prediction_residual(samples, ())
    else:
        # Find best fixed predictor order for the given samples
        residuals = [
            prediction_residual(samples, coefficients)
            for coefficients in FIXED_PREDICTOR_COEFFICIENTS
        ]

        # mypy infers the type of total_error is Any, not sure why.
        # Explicitly declaring the type for the moment, but might be an hint
        # for a possible bug.
        # TODO: investigate further
        total_error: list[int] = [sum(abs(r) for r in rs) for rs in residuals]

        order = min(range(len(total_error)), key=lambda x: total_error[x])
        warmup = samples[:order]
        residual = residuals[order]

    header = SubframeHeader(SubframeTypeFixed(order=order), 0)
    subframe = SubframeFixed(warmup, residual)

    return (header, subframe)


def encode_subframe_lpc(
        samples: list[int],
        lpc_order: range,
        precision: int
) -> tuple[SubframeHeader, SubframeLPC]:
    # 1. Samples apodization to improve Levinson-Durbin algorithm stability
    windowed = [x * w for x, w in zip(samples, tukey(len(samples), 0.5))]

    # 2. Compute autocorrelation
    autocorrs = [autocorrelation(windowed, i) for i in range(lpc_order.stop)]

    # 3. Get floating point coefficients
    coefficients = [levinson_durbin(autocorrs[:i])
                    for i in range(2, lpc_order.stop + 1)]

    # 4. Quantize coefficients. Get integer coefficients and shift
    quantized_coefficients_and_shift = [
        quantize_lpc_coefficients(coefficient, precision)
        for coefficient in coefficients
    ]

    quantized_coefficients = [x[0] for x in quantized_coefficients_and_shift]
    shifts = [x[1] for x in quantized_coefficients_and_shift]

    # 5. Compute residuals for each set of coefficients
    residuals = [
        prediction_residual(samples, coeffs, shift)
        for coeffs, shift in quantized_coefficients_and_shift
    ]

    # 6. Compute total error for each set of coefficients.
    #
    # mypy infers the type of total_error is Any, not sure why.
    # Explicitly declaring the type for the moment, but might be an hint
    # for a possible bug.
    # TODO: investigate further
    total_error: list[int] = [sum(abs(r) for r in rs) for rs in residuals]

    # 7. Find best set of coefficients.
    # Beware: order is actually the index into coefficients,
    # quantized_coefficients, shifts and residuals, not really the order.
    # The actual order is order + 1.
    order = min(range(len(total_error)), key=lambda x: total_error[x])

    coeffs = quantized_coefficients[order]
    shift = shifts[order]
    warmup = samples[:order + 1]
    residual = residuals[order]

    header = SubframeHeader(SubframeTypeLPC(order=order + 1), 0)
    subframe = SubframeLPC(
        warmup=warmup,
        precision=precision,
        shift=shift,
        coefficients=coeffs,
        residual=residual
    )

    return (header, subframe)


def tukey(
        n: int,  # window size
        r: float  # cosine fraction, often referred to as alpha
) -> list[float]:
    # Given that I found way too many definitions for the Tukey window,
    # sometimes even supposedly not correct (thanks Wikipedia), the following
    # code is shamelessly stolen from libflac, window.c, FLAC__window_tukey.
    nr = floor(r / 2.0 * n) - 1

    # Start with a rectangle window...
    xs = [1.0] * n

    # ... replace ends with Hann window
    for i in range(nr + 1):
        xs[i] = 0.5 - 0.5 * cos(pi * i / nr)
        xs[n - nr - 1 + i] = 0.5 - 0.5 * cos(pi * (i + nr) / nr)

    return xs


def autocorrelation(
        samples: list[float],
        lag: int
) -> float:
    return sum(
        samples[j] * samples[j + lag]
        for j in range(len(samples) - lag - 1)
    )


def levinson_durbin(
        xs: list[float],  # autocorrelation values
) -> list[float]:
    assert len(xs) > 1

    order = len(xs) - 1
    coefs = [0.0] * (order + 1)
    coefs[0] = 1.0
    error = xs[0]

    for k in range(order):
        lambda_ = 0.0

        for j in range(k + 1):
            lambda_ -= coefs[j] * xs[k + 1 - j]

        lambda_ /= error

        for n in range((k + 1) // 2 + 1):
            temp = coefs[k + 1 - n] + lambda_ * coefs[n]
            coefs[n] = coefs[n] + lambda_ * coefs[k + 1 - n]
            coefs[k + 1 - n] = temp

        error *= 1.0 - lambda_ ** 2

    # return coefs[1:], error
    return coefs[1:]


def quantize_lpc_coefficients(
        coefficients: list[float],
        precision: int,
) -> tuple[list[int], int]:  # coefficients at list of integers, shift
    # Code stolen from libFLAC, lpc.c, FLAC__lpc_quantize_coefficients.
    # Beware: very not Pythonic code follows.
    # TODO: refactor this mess.

    # 5 is the min precision value that can be used
    # TODO: extract this constant?
    assert precision >= 5

    # 1. Compute the shift value
    coef_max = max([abs(x) for x in coefficients])
    assert coef_max > 0.0

    # 5 is the size of the shift field in the binary representation
    # TODO: extract these kind of constants?
    shift_max = (1 << (5 - 1)) - 1
    shift_min = - (1 << (5 - 1))

    shift = precision - floor(log2(coef_max)) - 2

    if shift > shift_max:
        shift = shift_max
    elif shift < shift_min:
        assert False

    # 2. Compute the quantized coefficients
    qlp_max = (1 << (precision - 1)) - 1
    qlp_min = - (1 << (precision - 1))

    quantized_coefficients = []
    error = 0.0

    if shift >= 0:
        for coefficient in coefficients:
            error += coefficient * (1 << shift)
            q = clamp(round(error), qlp_min, qlp_max)
            error -= q  # BEWARE: operaton between float and int, even if safe
            quantized_coefficients.append(q)
    else:
        # Negative shift is very rave but possible. Due to a design flaw in
        # FLAC, negative shift is not allowed in the decoder, so it must
        # specially handled by scaling down the coefficients.
        nshift = - shift
        for coefficient in coefficients:
            error += coefficient * (1 << nshift)
            q = clamp(round(error), qlp_min, qlp_max)
            error -= q  # BEWARE: operaton between float and int, even if safe
        shift = 0

    return quantized_coefficients, shift


def prediction_residual(
        samples: list[int],
        coefficients: Sequence[int],
        shift: int = 0
) -> list[int]:
    order = len(coefficients)
    return [
        (samples[i]
         - (sum(samples[i - 1 - j] * c for j, c in enumerate(coefficients))
            >> shift))
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
        partition_order_range: range
):
    residual = encode_residual(
        subframe.residual,
        block_size,
        sample_size,
        subframe.order,
        partition_order_range
    )

    for s in subframe.warmup:
        put.uint(s, sample_size)  # signed int actually
    put_residual(put, residual, sample_size)


def _put_subframe_lpc(
        put: Put,
        subframe: SubframeLPC,
        block_size: int,
        sample_size: int,
        partition_order_range: range
):
    residual = encode_residual(
        subframe.residual,
        block_size,
        sample_size,
        subframe.order,
        partition_order_range
    )

    for s in subframe.warmup:
        put.uint(s, sample_size)  # signed int actually

    assert subframe.precision - 1 != 0b1111
    put.uint(subframe.precision - 1, 4)

    put.uint(subframe.shift, 5)  # signed int actually

    for c in subframe.coefficients:
        put.uint(c, subframe.precision)  # signed int actually

    put_residual(put, residual, sample_size)


# -----------------------------------------------------------------------------

def encode_residual(
        samples: list[int],
        block_size: int,
        sample_size: int,
        predictor_order: int,
        partition_order_range: range
) -> Residual:
    samples_ = [zigzag_encode(x) for x in samples]

    partitions = rice_partitions(
        samples_,
        block_size,
        predictor_order,
        partition_order_range
    )

    coding_method = (RiceCodingMethod.Rice4Bit
                     if all(p.parameter <= 14 for p in partitions)
                     else RiceCodingMethod.Rice5Bit)

    return Residual(coding_method, partitions)


def rice_partitions(
        samples: list[int],  # zig-zag encoded samples
        block_size: int,
        predictor_order: int,
        partition_order_range: range
) -> list[RicePartition]:
    # The partition order MUST be so that the block size is evenly divisible by
    # the number of partitions. The partition order also MUST be so that the
    # (block size >> partition order) is larger than the predictor order.
    candidate_orders = [
        o for o in partition_order_range
        if (block_size % (1 << o) == 0 and (block_size >> o) > predictor_order)
    ]

    assert len(candidate_orders) > 0

    # Partition configuration (how samples are grouped in partitions, list of
    # list of samples) for each candidate partition order
    partitions = [
        split_samples_in_partitions(samples, o, block_size, predictor_order)
        for o in candidate_orders
    ]

    # For each partition configuration, the size and the rice parameters for
    # partition.
    sizes_and_parameters = [
        [estimate_rice_partition_size_and_parameter(p) for p in ps]
        for ps in partitions
    ]

    sizes = [sum(x[0] for x in xs) for xs in sizes_and_parameters]
    parameters = [[x[1] for x in xs] for xs in sizes_and_parameters]

    best_configuration = min(zip(sizes, parameters, partitions),
                             key=lambda x: x[0])

    return [
        RicePartition(parameter, samples)
        for (parameter, samples)
        in zip(best_configuration[1], best_configuration[2])
    ]


def split_samples_in_partitions(
        samples: list[int],
        partition_order: int,
        block_size: int,
        predictor_order: int
) -> list[list[int]]:
    "Return the given samples grouped in partitions."
    p0_samples_ = (block_size >> partition_order) - predictor_order
    ps_samples_ = block_size >> partition_order

    p0_samples = samples[:p0_samples_]
    ps_samples = group(samples[p0_samples_:], ps_samples_)

    return [p0_samples, *ps_samples]


def estimate_rice_partition_size_and_parameter(
        samples: list[int]
) -> tuple[int, int]:
    parameter = find_rice_parameter(samples)
    parameter_size = 5 if parameter > 14 else 4
    size = sum(rice_size(s, parameter) for s in samples)

    partition_size = (
        4  # number of bits to encode the partition order
        + parameter_size  # number of bits to encode the rice parameter
        + size
    )

    return (partition_size, parameter)


def find_rice_parameter(
        samples: list[int],  # zig-zag encoded samples
) -> int:
    # Code stolen from libFLAC, stream_encoder.c, set_partitioned_rice.
    # The following comment is present, introducing some code that has been
    # disabled and some more code that I have not spent enough time to fully
    # understand. What I'm doing is blindly implementing what the comment
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

    # TODO: assert that found value is <= 14 for sample_size <= 16 or that it
    #       is <= 30 otherwise.
    return floor(log2(sum(samples)/len(samples)))


def rice_size(x: int, parameter: int) -> int:
    "Size of rice-encoded number in bits"
    m = 1 << parameter
    q = x // m
    return q + 1 + parameter


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
