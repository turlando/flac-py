from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from functools import reduce
from operator import add
from pathlib import Path
from timeit import default_timer as timer
from typing import Iterator
from wave import open as wave_open

from flac.utils import argparse_range, group
from flac.decoder import decode
from flac.encoder import EncoderParameters, encode


# -----------------------------------------------------------------------------

ACTION_ENCODE = 'encode'
ACTION_DECODE = 'decode'

DEFAULT_BLOCK_SIZE = 4608
DEFAULT_MAX_LPC_ORDER = 12
DEFAULT_QLP_COEFF_PRECISION = 5
DEFAULT_RICE_PARTITION_ORDER = '5'


# -----------------------------------------------------------------------------

def cmd_decode(path_in: Path, path_out: Path):
    with (
        path_in.open('rb') as f,
        wave_open(str(path_out), mode='wb') as w
    ):
        (sample_rate, sample_size, channels, frames, samples) = decode(f)

        # Python's wave module doesn't handle sample sizes that are not
        # multiple of a byte.
        assert sample_size % 8 == 0

        w.setframerate(sample_rate)
        w.setsampwidth(sample_size // 8)
        w.setnchannels(channels)
        w.setnframes(frames)

        time_start = timer()

        for s in samples:
            bs = [channel.to_bytes(sample_size // 8,
                                   byteorder='little',
                                   signed=True)
                  for channel in s]
            w.writeframes(reduce(add, bs))

        time_end = timer()

        delta = '{0:.6g}'.format(time_end - time_start)
        print(f"Decoding completed in {delta} seconds")


def cmd_encode(
        path_in: Path,
        path_out: Path,
        block_size: int,
        max_lpc_order: int,
        qlp_coeffs_precision: int,
        rice_partition_order: range
):
    with (
        wave_open(str(path_in), mode='rb') as f_in,
        path_out.open('wb') as f_out
    ):
        parameters = EncoderParameters(
            block_size=block_size,
            lpc_order=range(max_lpc_order + 1),
            qlp_precision=qlp_coeffs_precision,
            rice_partition_order=rice_partition_order
        )

        sample_rate = f_in.getframerate()
        sample_size_bytes = f_in.getsampwidth()
        channels = f_in.getnchannels()
        frames = f_in.getnframes()

        def samples() -> Iterator[list[int]]:
            while True:
                xs = f_in.readframes(1)

                if xs == b'':
                    return

                assert len(xs) == channels * sample_size_bytes

                yield [int.from_bytes(x, byteorder='little', signed=True)
                       for x in group(xs, channels)]

        time_start = timer()

        for bs in encode(
                sample_rate,
                sample_size_bytes * 8,
                channels,
                frames,
                samples(),
                parameters
        ):
            f_out.write(bs)

        time_end = timer()

        delta = '{0:.6g}'.format(time_end - time_start)
        print(f"Encoding completed in {delta} seconds")


# -----------------------------------------------------------------------------

def make_argument_parser():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    # -------------------------------------------------------------------------

    action = parser.add_subparsers(
        title='action',
        dest='action',
        required=True
    )

    # -------------------------------------------------------------------------

    decode = action.add_parser(
        ACTION_DECODE,
        formatter_class=ArgumentDefaultsHelpFormatter
    )

    decode.add_argument('infile', type=Path)
    decode.add_argument('outfile', type=Path)

    # -------------------------------------------------------------------------

    encode = action.add_parser(
        ACTION_ENCODE,
        formatter_class=ArgumentDefaultsHelpFormatter
    )

    encode.add_argument('infile', type=Path)
    encode.add_argument('outfile', type=Path)

    encode.add_argument(
        '-b', '--block-size',
        type=int,
        default=DEFAULT_BLOCK_SIZE,
        help=(
            "Blocksize in samples. "
            "For subset streams this must be <= 4608 if the samplerate <= "
            "48kHz. For  subset streams with higher samplerates it must be "
            "<= 16384."
        ),
        metavar='N'
    )

    encode.add_argument(
        '-l', '--max-lpc-order',
        type=int,
        default=DEFAULT_MAX_LPC_ORDER,
        help=(
            "Specifies  the  maximum LPC order. This number must "
            "be <= 32. For subset streams, it must be <= 12 if the "
            "sample rate is <=48kHz."
        ),
        metavar='N'
    )

    encode.add_argument(
        '-q', '--qlp-coeff-precision',
        type=int,
        default=DEFAULT_QLP_COEFF_PRECISION,
        help=(
            "Precision of the quantized linear-predictor coefficients. "
            "(min is 5)"
        ),
        metavar='N'
    )

    encode.add_argument(
        '-r', '--rice-partition-order',
        type=argparse_range,
        default=DEFAULT_RICE_PARTITION_ORDER,
        help=(
            "[min,]max residual partition order (0..15). min defaults to 0 if "
            "unspecified."
        ),
        metavar='[M,]N'
    )

    # -------------------------------------------------------------------------

    return parser


def main():
    parser = make_argument_parser()
    args = parser.parse_args()

    if args.action == ACTION_DECODE:
        cmd_decode(args.infile, args.outfile)

    if args.action == ACTION_ENCODE:
        cmd_encode(
            args.infile,
            args.outfile,
            args.block_size,
            args.max_lpc_order,
            args.qlp_coeff_precision,
            args.rice_partition_order
        )


# -----------------------------------------------------------------------------

if __name__ == '__main__':
    main()
