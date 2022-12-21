from collections.abc import Iterator
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


def write_wav(
        writer: Writer,
        sample_rate: int,
        sample_size: int,
        channels: int,
        samples_count: int,
        samples: Iterator[int]
):
    # FIXME: this encoder might not work with odd bytes sample_size
    writer.write_bytes(b'RIFF')

    sample_bytes = sample_size // 8
    size = samples_count * channels * sample_bytes

    writer.write_bytes((size + 36).to_bytes(4, 'little'))

    writer.write_bytes(b'WAVE')

    # Format Chunk ID
    writer.write_bytes(b'fmt ')

    # Chunk data size: 16 + extra format (0)
    writer.write_bytes((16).to_bytes(4, byteorder='little'))

    # Compression code: 0x0001 = PCM/uncompressed
    writer.write_bytes((1).to_bytes(2, byteorder='little'))

    # Number of channels
    writer.write_bytes(channels.to_bytes(2, byteorder='little'))

    # Sample rate
    writer.write_bytes(sample_rate.to_bytes(4, byteorder='little'))

    # Average bytes per second
    abps = sample_rate * channels * sample_bytes
    writer.write_bytes(abps.to_bytes(4, byteorder='little'))

    # Block align
    align = channels * sample_bytes
    writer.write_bytes(align.to_bytes(2, byteorder='little'))

    # Significant bits per sample
    writer.write_bytes(sample_size.to_bytes(2, byteorder='little'))

    # Data Chunk ID
    writer.write_bytes(b'data')

    # Chunk size
    writer.write_bytes(size.to_bytes(4, byteorder='little'))

    for sample in samples:
        b = sample.to_bytes(sample_bytes, byteorder='little', signed=True)
        writer.write_bytes(b)


if __name__ == '__main__':
    main(argv)
