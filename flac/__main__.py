from pathlib import Path
from sys import argv

from flac.decoder import decode
from flac.binary import Reader


def main(args):
    path = Path(args[1])
    with path.open('rb') as f:
        decode(Reader(f))


if __name__ == '__main__':
    main(argv)
