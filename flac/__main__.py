from pathlib import Path
from sys import argv
from timeit import default_timer as timer

from flac.binary import Reader, Writer
from flac.decoder import (
    MetadataBlockType,
    consume_magic, read_metadata_block_header, read_metadata_block_streaminfo,
    skip_metadata, read_frames, decode_frame
)


def main(args):
    path_in = Path(args[1])
    path_out = Path(args[2])

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
        print(f"    Sample rate: {streaminfo.sample_rate}")
        print(f"    Sample size: {streaminfo.sample_size}")
        print(f"    Channels...: {streaminfo.channels}")

        if streaminfo_header.last is False:
            skip_metadata(r)

        time_start = timer()

        for frame in read_frames(r, streaminfo):
            frames = decode_frame(frame)
            for channels in zip(*frames):
                for sample in channels:
                    w.write(sample, streaminfo.sample_size)

        time_end = timer()

        delta = '{0:.6g}'.format(time_end - time_start)
        print(f"Decoding completed in {delta} seconds")

        print("Play with: "
              f"ffplay -f s{streaminfo.sample_size}be "
              f"-ar {streaminfo.sample_rate} "
              f"-ac {streaminfo.channels} {path_out}")


if __name__ == '__main__':
    main(argv)
