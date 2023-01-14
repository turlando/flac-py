from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from functools import reduce
from operator import add
from pathlib import Path
from timeit import default_timer as timer
from typing import Iterator
from wave import open as wave_open

from flac.utils import group
from flac.decoder import decode
from flac.encoder import EncoderParameters, encode


# -----------------------------------------------------------------------------

ACTION_ENCODE = 'encode'
ACTION_DECODE = 'decode'

DEFAULT_BLOCK_SIZE = 4608


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
        block_size: int
):
    with (
        wave_open(str(path_in), mode='rb') as f_in,
        path_out.open('wb') as f_out
    ):
        parameters = EncoderParameters(block_size=block_size)

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
        help="blocksize in samples"
    )

    # -------------------------------------------------------------------------

    return parser


if __name__ == '__main__':
    parser = make_argument_parser()
    args = parser.parse_args()

    if args.action == ACTION_DECODE:
        cmd_decode(args.infile, args.outfile)

    if args.action == ACTION_ENCODE:
        cmd_encode(args.infile, args.outfile, args.block_size)
