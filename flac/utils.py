from argparse import Action
from enum import Enum
from itertools import islice
from typing import Iterator, Sequence, TypeVar


T = TypeVar('T')


def batch(it: Iterator[T], n: int):
    """
    Batch data into tuples of length n. The last batch may be shorter.
    >>> [x for x in batch(iter('ABCDEFG'), 3)]
    [('A', 'B', 'C'), ('D', 'E', 'F'), ('G',)]
    """
    if n < 1:
        raise ValueError('n must be greater than zero')
    while (batch := tuple(islice(it, n))):
        yield batch


def group(xs: Sequence[T], n: int) -> list[Sequence[T]]:
    """
    >>> group([1, 2, 3, 4, 5, 6], 2)
    [[1, 2], [3, 4], [5, 6]]
    """
    return [xs[i:i+n] for i in range(0, len(xs), n)]


class EnumAction(Action):
    def __init__(self, **kwargs):
        type = kwargs.pop("type", None)

        if type is None:
            raise ValueError("type must be an Enum")
        if not issubclass(type, Enum):
            raise TypeError("type must be an Enum")

        kwargs.setdefault("choices", tuple(e.value for e in type))
        super(EnumAction, self).__init__(**kwargs)
        self._enum = type

    def __call__(self, parser, namespace, values, option_string=None):
        value = self._enum(values)
        setattr(namespace, self.dest, value)
