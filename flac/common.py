from dataclasses import dataclass
from enum import auto
from typing import Optional
from flac.utils import Enum, invert_dict, log2i


# -----------------------------------------------------------------------------

MAGIC = b'fLaC'
FRAME_SYNC_CODE = 0b111111111111100

CRC8_POLYNOMIAL = 0b1_00000111  # x^8 + x^2 + x^1 + x^0
CRC16_POLYNOMIAL = 0b1_10000000_00000101  # x^16 + x^15 + x^2 + x^0

FIXED_PREDICTOR_COEFFICIENTS = (
    (),
    (1,),
    (2, -1),
    (3, -3, 1),
    (4, -6, 4, -1)
)


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


BLOCK_SIZE_ENCODING = {
    192: 0b0001,

    # 0b0010 - 0b0101: 144 * (2^v)
    # i.e. 576, 1152, 2304 or 4608
    576: 0b0010,
    1152: 0b0011,
    2304: 0b0100,
    4608: 0b0101,

    # 0b1000 - 0b1111: 2^v
    # i.e. 256, 512, 1024, 2048, 4096, 8192, 16384 or 32768
    256: 0b1000,
    512: 0b1001,
    1024: 0b1010,
    2048: 0b1011,
    4096: 0b1100,
    8192: 0b1101,
    16384: 0b1110,
    32768: 0b1111
}


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


SAMPLE_RATE_VALUE_ENCODING = {
    SampleRateValue.V_88_2_kHz: 0b0001,
    SampleRateValue.V_176_4_kHz: 0b0010,
    SampleRateValue.V_192_kHz: 0b0011,
    SampleRateValue.V_8_kHz: 0b0100,
    SampleRateValue.V_16_kHz: 0b0101,
    SampleRateValue.V_22_05_kHz: 0b0110,
    SampleRateValue.V_24_kHz: 0b0111,
    SampleRateValue.V_32_kHz: 0b1000,
    SampleRateValue.V_44_1_kHz: 0b1001,
    SampleRateValue.V_48_kHz: 0b1010,
    SampleRateValue.V_96_kHz: 0b1100
}

SAMPLE_RATE_VALUE_DECODING = invert_dict(SAMPLE_RATE_VALUE_ENCODING)


# -----------------------------------------------------------------------------

class Channels(Enum):
    M = auto()
    L_R = auto()
    L_R_C = auto()
    FL_FR_BL_BR = auto()
    FL_FR_FC_BL_BR = auto()
    FL_FR_FC_LFE_BL_BR = auto()
    FL_FR_FC_LFE_BC_SL_SR = auto()
    FL_FR_FC_LFE_BL_BR_SL_SR = auto()
    L_S = auto()
    S_R = auto()
    M_S = auto()

    @property
    def count(self) -> int:
        return CHANNELS_COUNT[self]

    @property
    def decorrelation_bit(self) -> list[int]:
        # Side channel has one extra bit in sample_size
        match self:
            case self.L_S:
                return [0, 1]
            case self.S_R:
                return [1, 0]
            case self.M_S:
                return [0, 1]
            case _:
                return [0] * self.count


CHANNELS_ENCODING = {
    Channels.M: 0b0000,
    Channels.L_R: 0b0001,
    Channels.L_R_C: 0b0010,
    Channels.FL_FR_BL_BR: 0b0011,
    Channels.FL_FR_FC_BL_BR: 0b0100,
    Channels.FL_FR_FC_LFE_BL_BR: 0b0101,
    Channels.FL_FR_FC_LFE_BC_SL_SR: 0b0110,
    Channels.FL_FR_FC_LFE_BL_BR_SL_SR: 0b0111,
    Channels.L_S: 0b1000,
    Channels.S_R: 0b1001,
    Channels.M_S: 0b1010
}

CHANNELS_DECODING = invert_dict(CHANNELS_ENCODING)

CHANNELS_COUNT = {
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
}


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


SampleSize = SampleSizeFromStreaminfo | SampleSizeValue


SAMPLE_SIZE_ENCODING = {
    SampleSizeValue.V_8: 0b001,
    SampleSizeValue.V_12: 0b010,
    SampleSizeValue.V_16: 0b100,
    SampleSizeValue.V_20: 0b101,
    SampleSizeValue.V_24: 0b110,
    SampleSizeValue.V_32: 0b111
}

SAMPLE_SIZE_DECODING = invert_dict(SAMPLE_SIZE_ENCODING)


# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class FrameHeader:
    blocking_strategy: BlockingStrategy
    block_size: int
    sample_rate: Optional[int]
    channels: Channels
    sample_size: Optional[int]
    coded_number: int
    crc: Optional[int] = None


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


# -----------------------------------------------------------------------------

class RiceCodingMethod(Enum):
    Rice4Bit = 4
    Rice5Bit = 5


@dataclass
class RicePartition:
    encoding_parameter: int
    residual: list[int]

    def __repr__(self):
        return ("RicePartition("
                f"encoding_parameter={self.encoding_parameter}, "
                f"samples_count={len(self.residual)}"
                ")")


@dataclass
class Residual:
    coding_method: RiceCodingMethod
    partitions: list[RicePartition]

    @property
    def partition_order(self) -> int:
        return log2i(len(self.partitions))

    def __repr__(self):
        return ("Residual("
                f"coding_method={self.coding_method}, "
                f"partition_order={self.partition_order}, "
                f"partitions={self.partitions}"
                ")")
