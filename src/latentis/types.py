from typing import Callable, Tuple

import torch

# try:
#     # be ready for 3.10 when it drops
#     from enum import StrEnum
# except ImportError:
#     from backports.strenum import StrEnum

TransformType = Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]
