from argparse import ArgumentParser
from enum import Enum
from pathlib import Path
from sys import argv
from timeit import default_timer as timer

from flac.utils import EnumAction
from flac.binary import Reader, Writer
from flac.common import MetadataBlockType
from flac.wave import write_wav
from flac.decoder import (
    consume_magic, read_metadata_block_header, read_metadata_block_streaminfo,
    skip_metadata, read_frames, decode_frame
)


class Action(Enum):
    Decode = 'decode'
    Encode = 'encode'


def decode(path_in: Path, path_out: Path):
    with (
        path_in.open('rb') as f_in,
        path_out.open('wb') as f_out
    ):
        r = Reader(f_in)
        w = Writer(f_out)

        consume_magic(r)

        streaminfo_header = read_metadata_block_header(r)
        assert streaminfo_header.type == MetadataBlockType.Streaminfo

        streaminfo = read_metadata_block_streaminfo(r)

        print("Stream info:")
        print(f"    Samples....: {streaminfo.samples}")
        print(f"    Sample rate: {streaminfo.sample_rate}")
        print(f"    Sample size: {streaminfo.sample_size}")
        print(f"    Channels...: {streaminfo.channels}")

        if streaminfo_header.last is False:
            skip_metadata(r)

        def samples():
            for frame in read_frames(r, streaminfo):
                frames = decode_frame(frame)
                for channels in zip(*frames):
                    for sample in channels:
                        yield sample

        time_start = timer()

        write_wav(
            w,
            streaminfo.sample_rate,
            streaminfo.sample_size,
            streaminfo.channels,
            streaminfo.samples,
            samples()
        )

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
            decode(args.infile, args.outfile)
        case Action.Encode:
            pass
