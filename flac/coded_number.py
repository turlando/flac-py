from flac.binary import extract, mask
from functools import reduce


# -----------------------------------------------------------------------------

def encode(x: int) -> bytes:
    assert 0 <= x.bit_length() <= 36

    # Total number of bytes, including the first byte.
    size = required_bytes(x)

    if size == 1:
        return x.to_bytes(1, byteorder='big')

    bs = [(x >> 6 * i) & mask(6) for i in reversed(range(0, size))]

    return bytes([
        mask(size) << (8 - size) | bs[0],
        *[(0b10 << 6) | b for b in bs[1:]]
    ])


def required_bytes(x: int) -> int:
    match x.bit_length():
        case n if 0 <= n <= 7:
            return 1
        case n if 8 <= n <= 11:
            return 2
        case n if 12 <= n <= 16:
            return 3
        case n if 17 <= n <= 21:
            return 4
        case n if 22 <= n <= 26:
            return 5
        case n if 27 <= n <= 31:
            return 6

    raise ValueError(f"Cannot encode coded number: {x}")


# -----------------------------------------------------------------------------

def decode(bs: bytes) -> int:
    size = following_bytes(bs[0]) + 1
    assert size == len(bs)

    if size == 1:
        return bs[0]

    x0 = extract(bs[0], 8, 2 + (size - 1), 8)
    xs = reduce(lambda acc, x: (acc << 6) | extract(x, 8, 2, 8), bs[1:], 0)

    return (x0 << (size - 1) * 6) | xs


def following_bytes(x: int) -> int:
    if x >= 0b11111110:
        return 6
    if x >= 0b11111100:
        return 5
    if x >= 0b11111000:
        return 4
    if x >= 0b11110000:
        return 3
    if x >= 0b11100000:
        return 2
    if x >= 0b11000000:
        return 1
    else:
        return 0
