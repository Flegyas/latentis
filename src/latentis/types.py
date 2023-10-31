# try:
#     # be ready for 3.10 when it drops
#     from enum import StrEnum
# except ImportError:
#     from backports.strenum import StrEnum
from __future__ import annotations

from typing import Callable, Mapping, Union

import torch

from latentis.space import LatentSpace

ProjectionFunc = Callable[[torch.Tensor, torch.Tensor], Mapping[str, torch.Tensor]]

Space = Union[LatentSpace, torch.Tensor]
