from collections.abc import Iterator
from flac.binary import Writer


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
