# flac-py

Self-contained, zero dependencies, pure Python, extremely slow implementation
of a FLAC encoder and decoder.

The purpose of this software is to explore and understand the FLAC codec and
file format. This is a didactic tool and thus not to be intended to be used
in real-world scenarios.

It is not supposed to be fast or efficient, but it is supposed to be clear,
understandable, idiomatic and correct.

Python 3.10 or better is required.

## Installation

Given the absence of external dependencies, this software can be run straight
from the directory containing this document with the following command.

```sh
python -m flac
```

It is otherwise possible to install this package using pip with the following
command.

```sh
pip install .
```

It is nonetheless suggested to install this package inside a Python virtual
environment. To do so the `venv` module (or equivalent) must be available.
`venv` is included in the default Python distribution, but some packagers such
as Debian provide it in a separate package.

```sh
# Create a new virtual environment in the env directory.
python -m venv env

# Activate the virtual environment.
# The following command works for bash-like shells. csh, fish and Windows' cmd
# have their own script.
source env/bin/activate

# Now flac-py is available to the shell.

# Deactivate the virtual environment.
deactivate
```

## Usage

```
usage: flac-py [-h] {decode,encode} ...
flac-py: error: the following arguments are required: action
```

```
usage: flac-py decode [-h] infile outfile

positional arguments:
  infile
  outfile

options:
  -h, --help  show this help message and exit
```

```
usage: flac-py encode [-h] [-b N] [-l N] [-q N] [-r [M,]N] infile outfile

positional arguments:
  infile
  outfile

options:
  -h, --help            show this help message and exit
  -b N, --block-size N  Blocksize in samples. For subset streams this must be
                        <= 4608 if the samplerate <= 48kHz. For subset streams
                        with higher samplerates it must be <= 16384. (default:
                        4608)
  -l N, --max-lpc-order N
                        Specifies the maximum LPC order. This number must be
                        <= 32. For subset streams, it must be <= 12 if the
                        sample rate is <=48kHz. (default: 12)
  -q N, --qlp-coeff-precision N
                        Precision of the quantized linear-predictor
                        coefficients. (min is 5) (default: 5)
  -r [M,]N, --rice-partition-order [M,]N
                        [min,]max residual partition order (0..15). min
                        defaults to 0 if unspecified. (default: 5)
```

## Development

It is suggested to install this package inside a virtual environment as editable,
alongside all the development dependencies (namely a test runner, a linter and
a type checker). To do so run the following command inside a virtual environment.

```sh
pip install -e '.[dev]'
```

To make sure the code is in good shape run the following command.

```sh
flake8
```

To make sure the code type checks run the following command.

```sh
mypy -p flac
```

To run the unit tests run the following command.

```sh
pytest
```

## Known issues and TODOs

* While encoding with arbitrary LPC coefficients (SubframeLPC) works, it looks
  like it is not determining the best coefficients, yielding files bigger in
  size compared to using fixed coefficients (SubframeFixed) only.
* Stereo decorrelation is not implemented yet.
* It would be nice to have a tool providing information about a FLAC file,
  as `flac -a` does.
* All the TODOs spread around the code.


## License

Copyright (C) 2023 Tancredi Orlando

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, version 3 of the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
