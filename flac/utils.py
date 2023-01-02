from enum import Enum
from argparse import Action


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
