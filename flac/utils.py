from argparse import Action
from enum import Enum as _Enum
from itertools import islice
from typing import Iterator, TypeVar, overload


T = TypeVar('T')
K = TypeVar('K')
V = TypeVar('V')


def batch(it: Iterator[T], n: int) -> Iterator[list[T]]:
    """
    Batch data into tuples of length n. The last batch may be shorter.
    >>> [x for x in batch(iter('ABCDEFG'), 3)]
    [('A', 'B', 'C'), ('D', 'E', 'F'), ('G',)]
    """
    if n < 1:
        raise ValueError('n must be greater than zero')
    while (batch := list(islice(it, n))):
        yield batch


@overload
def group(xs: bytes, n: int) -> list[bytes]:
    ...


@overload
def group(xs: list[T], n: int) -> list[list[T]]:
    ...


def group(xs, n):
    """
    >>> group([1, 2, 3, 4, 5, 6], 2)
    [[1, 2], [3, 4], [5, 6]]
    """
    return [xs[i:i+n] for i in range(0, len(xs), n)]


def invert_dict(d: dict[K, V]) -> dict[V, K]:
    return {v: k for k, v in d.items()}


def log2i(x: int) -> int:
    "Base 2 integer logarithm"
    assert x > 0
    res = 0
    while x > 0:
        x >>= 1
        res += 1
    return res - 1


def zigzag_decode(x: int) -> int:
    return (x >> 1) ^ -(x & 1)


def zigzag_encode(x: int, n: int) -> int:
    assert n > 0
    assert (-1 << (n - 1)) < x < (1 << n) - 1
    return (x >> (n - 1)) ^ (x << 1)


class Enum(_Enum):
    @classmethod
    def values(cls):
        return set(x.value for x in cls.__members__.values())


class EnumAction(Action):
    def __init__(self, **kwargs):
        type = kwargs.pop("type", None)

        if type is None:
            raise ValueError("type must be an Enum")
        if not issubclass(type, _Enum):
            raise TypeError("type must be an Enum")

        kwargs.setdefault("choices", tuple(e.value for e in type))
        super(EnumAction, self).__init__(**kwargs)
        self._enum = type

    def __call__(self, parser, namespace, values, option_string=None):
        value = self._enum(values)
        setattr(namespace, self.dest, value)
