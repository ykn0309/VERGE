"""VERGE requires Python 3.10 or newer."""

import sys

if sys.version_info < (3, 10):
    raise RuntimeError(
        "VERGE requires Python >=3.10; recommended: /opt/anaconda3/bin/python (3.12)"
    )
