from dataclasses import dataclass
from enum import Enum
from typing import Optional


# -----------------------------------------------------------------------------

MAGIC = b'fLaC'


# -----------------------------------------------------------------------------

class MetadataBlockType(Enum):
    Streaminfo = 0
    Padding = 1
    Application = 2
    Seektable = 3
    VorbisComment = 4
    Cuesheet = 5
    Picture = 6


@dataclass(frozen=True)
class MetadataBlockHeader:
    last: bool
    type: MetadataBlockType
    length: int


# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class Streaminfo:
    min_block_size: int
    max_block_size: int
    min_frame_size: int
    max_frame_size: int
    sample_rate: int
    channels: int
    sample_size: int
    samples: int
    md5: bytes


# -----------------------------------------------------------------------------

class BlockingStrategy(Enum):
    Fixed = 0
    Variable = 1


# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class BlockSizeValue:
    size: int


@dataclass(frozen=True)
class BlockSizeUncommon8:
    pass


@dataclass(frozen=True)
class BlockSizeUncommon16:
    pass


BlockSize = BlockSizeValue | BlockSizeUncommon8 | BlockSizeUncommon16


# -----------------------------------------------------------------------------

class SampleRateFromStreaminfo:
    pass


class SampleRateValue(Enum):
    V_88_2_kHz = 88_200
    V_176_4_kHz = 176_400
    V_192_kHz = 192_000
    V_8_kHz = 8_000
    V_16_kHz = 16_000
    V_22_05_kHz = 22_050
    V_24_kHz = 24_000
    V_32_kHz = 32_000
    V_44_1_kHz = 44_100
    V_48_kHz = 48_000
    V_96_kHz = 96_000

    @classmethod
    def values(cls):
        return set(x.value for x in cls.__members__.values())

    def to_bin(self):
        return {
            self.V_88_2_kHz: 0b0001,
            self.V_176_4_kHz: 0b0010,
            self.V_192_kHz: 0b0011,
            self.V_8_kHz: 0b0100,
            self.V_16_kHz: 0b0101,
            self.V_22_05_kHz: 0b0110,
            self.V_24_kHz: 0b0111,
            self.V_32_kHz: 0b1000,
            self.V_44_1_kHz: 0b1001,
            self.V_48_kHz: 0b1010,
            self.V_96_kHz: 0b1100
        }[self]

    @classmethod
    def from_bin(cls, x: int):
        return {
            0b0001: cls.V_88_2_kHz,
            0b0010: cls.V_176_4_kHz,
            0b0011: cls.V_192_kHz,
            0b0100: cls.V_8_kHz,
            0b0101: cls.V_16_kHz,
            0b0110: cls.V_22_05_kHz,
            0b0111: cls.V_24_kHz,
            0b1000: cls.V_32_kHz,
            0b1001: cls.V_44_1_kHz,
            0b1010: cls.V_48_kHz,
            0b1100: cls.V_96_kHz
        }[x]

    def to_int(self):
        return self.value


class SampleRateUncommon8:
    pass


class SampleRateUncommon16:
    pass


class SampleRateUncommon16_10:
    pass


SampleRate = (
    SampleRateFromStreaminfo
    | SampleRateValue
    | SampleRateUncommon8
    | SampleRateUncommon16
    | SampleRateUncommon16_10
)


# -----------------------------------------------------------------------------

class Channels(Enum):
    M = 0b0000
    L_R = 0b0001
    L_R_C = 0b0010
    FL_FR_BL_BR = 0b0011
    FL_FR_FC_BL_BR = 0b0100
    FL_FR_FC_LFE_BL_BR = 0b0101
    FL_FR_FC_LFE_BC_SL_SR = 0b0110
    FL_FR_FC_LFE_BL_BR_SL_SR = 0b0111
    L_S = 0b1000
    S_R = 0b1001
    M_S = 0b1010

    def to_bin(self) -> int:
        return self.value

    @property
    def count(self) -> int:
        return {
            Channels.M: 1,
            Channels.L_R: 2,
            Channels.L_R_C: 3,
            Channels.FL_FR_BL_BR: 4,
            Channels.FL_FR_FC_BL_BR: 5,
            Channels.FL_FR_FC_LFE_BL_BR: 6,
            Channels.FL_FR_FC_LFE_BC_SL_SR: 7,
            Channels.FL_FR_FC_LFE_BL_BR_SL_SR: 8,
            Channels.L_S: 2,
            Channels.S_R: 2,
            Channels.M_S: 2
        }[self]

    @property
    def decorrelation_bit(self):
        # Side channel has one extra bit in sample_size
        match self:
            case self.L_S:
                return (0, 1)
            case self.S_R:
                return (1, 0)
            case self.M_S:
                return (0, 1)
            case _:
                return (0,) * self.count


# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class SampleSizeFromStreaminfo:
    pass


class SampleSizeValue(Enum):
    V_8 = 8
    V_12 = 12
    V_16 = 16
    V_20 = 20
    V_24 = 24
    V_32 = 32

    @classmethod
    def from_bin(cls, x: int):
        return {
            0b001: cls.V_8,
            0b010: cls.V_12,
            0b100: cls.V_16,
            0b101: cls.V_20,
            0b110: cls.V_24,
            0b111: cls.V_32
        }[x]

    def to_int(self):
        return self.value


SampleSize = SampleSizeFromStreaminfo | SampleSizeValue


# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class FrameHeader:
    blocking_strategy: BlockingStrategy
    block_size: int
    sample_rate: Optional[int]
    channels: Channels
    sample_size: Optional[int]
    coded_number: int
    crc: int


# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class SubframeTypeConstant:
    pass


@dataclass(frozen=True)
class SubframeTypeVerbatim:
    pass


@dataclass(frozen=True)
class SubframeTypeFixed:
    order: int


@dataclass(frozen=True)
class SubframeTypeLPC:
    order: int


SubframeType = (
    SubframeTypeConstant
    | SubframeTypeVerbatim
    | SubframeTypeFixed
    | SubframeTypeLPC
)


# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class SubframeHeader:
    type_: SubframeType
    wasted_bits: int


# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class SubframeConstant:
    sample: int
    block_size: int

    def __repr__(self):
        return "SubframeConstant()"


@dataclass(frozen=True)
class SubframeVerbatim:
    samples: list[int]

    def __repr__(self):
        return "SubframeVerbatim()"


@dataclass(frozen=True)
class SubframeFixed:
    warmup: list[int]
    residual: list[int]

    def __repr__(self):
        return f"SubframeFixed(order={len(self.warmup)})"


@dataclass(frozen=True)
class SubframeLPC:
    warmup: list[int]
    precision: int
    shift: int
    coefficients: list[int]
    residual: list[int]

    def __repr__(self):
        return f"SubframeLPC(order={len(self.warmup)})"


Subframe = SubframeConstant | SubframeVerbatim | SubframeFixed | SubframeLPC


# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class Frame:
    header: FrameHeader
    subframes: list[Subframe]
    crc: int
