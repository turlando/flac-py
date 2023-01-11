from argparse import ArgumentParser
from enum import Enum
from functools import reduce
from operator import add
from pathlib import Path
from sys import argv
from timeit import default_timer as timer
from typing import Iterator
from wave import open as wave_open

from flac.utils import EnumAction, group
from flac.decoder import decode
from flac.encoder import encode


class Action(Enum):
    Decode = 'decode'
    Encode = 'encode'


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


def cmd_encode(path_in: Path, path_out: Path):
    with (
        wave_open(str(path_in), mode='rb') as f_in,
        path_out.open('wb') as f_out
    ):
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
                samples()
        ):
            f_out.write(bs)

        time_end = timer()

        delta = '{0:.6g}'.format(time_end - time_start)
        print(f"Decoding completed in {delta} seconds")


def make_argument_parser():
    parser = ArgumentParser()
    parser.add_argument('action', action=EnumAction, type=Action)
    parser.add_argument('infile', type=Path)
    parser.add_argument('outfile', type=Path)
    return parser


if __name__ == '__main__':
    parser = make_argument_parser()
    args = parser.parse_args(None if argv[1:] else ['--help'])

    match args.action:
        case Action.Decode:
            cmd_decode(args.infile, args.outfile)
        case Action.Encode:
            cmd_encode(args.infile, args.outfile)
