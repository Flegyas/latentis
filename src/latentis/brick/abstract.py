import logging
from abc import abstractmethod
from typing import Any, Mapping, Optional

import torch
from torch import nn

from latentis.brick._errors import NotReversableBrickError

pylogger = logging.getLogger(__name__)

BrickState = Mapping[str, Any]


class Brick(nn.Module):
    def __init__(self, name: str) -> None:
        super().__init__()
        self._name: str = name
        self.saved_state: Optional[BrickState] = None

    @property
    def name(self) -> str:
        return self._name

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name})"

    def save_state(self, state: BrickState):
        self.saved_state = state

    def _check_state(self, state: Optional[BrickState] = None) -> None:
        if (self.saved_state is None) and (state is None):
            raise ValueError("Brick has no state saved or passed as argument")
        elif (self.saved_state is not None) and (state is not None):
            pylogger.warning("Brick has state saved and passed as argument")

    def _get_state(self, state: Optional[BrickState] = None) -> BrickState:
        self._check_state(state)
        return self.saved_state if state is None else state

    @abstractmethod
    def fit(self, *args, save: bool = True, **kwargs) -> BrickState:
        raise NotImplementedError

    @abstractmethod
    def forward(self, *args, state: Optional[BrickState] = None, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def reverse(self, *args, state: Optional[BrickState] = None, **kwargs) -> torch.Tensor:
        raise NotReversableBrickError
