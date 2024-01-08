from __future__ import annotations

from typing import TYPE_CHECKING, Union

import torch

try:
    # be ready for 3.10 when it drops
    from enum import StrEnum as PythonStrEnum
except ImportError:
    from backports.strenum import StrEnum as PythonStrEnum

if TYPE_CHECKING:
    from latentis.space import LatentSpace

    Space = Union[LatentSpace, torch.Tensor]

StrEnum = PythonStrEnum
