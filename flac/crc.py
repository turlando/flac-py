from functools import cache


@cache
def crc_table(n: int, generator: int) -> list[int]:
    t = []
    g = 1 << n | generator
    for divident in range(256):
        value = divident << (n - 8)
        for _ in range(8):
            value <<= 1
            if value & (1 << n) != 0:
                value ^= g
        t.append(value)
    return t


def crc8(bs: bytes, generator: int, initial_value: int = 0) -> int:
    table = crc_table(8, generator)
    crc = initial_value
    for b in bs:
        crc = table[b ^ crc]
    return crc


def crc16(bs: bytes, generator: int, initial_value: int = 0) -> int:
    table = crc_table(16, generator)
    crc = initial_value
    for b in bs:
        crc = ((crc << 8) & 0xFFFF) ^ table[(crc >> 8) ^ b]
    return crc
