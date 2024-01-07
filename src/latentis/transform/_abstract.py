from typing import Any, Mapping, Optional

import torch
from torch import nn

from latentis.transform.functional import ReverseFn, TransformFn, TransformResult

_PREFIX: str = "latentis_stat_"


class Transform(nn.Module):
    def __init__(
        self,
        transform_fn: TransformFn,
        name: Optional[str] = None,
        reverse_fn: Optional[ReverseFn] = None,
        fit_params: Optional[Mapping[str, Any]] = None,
        transform_params: Optional[Mapping[str, Any]] = None,
        reverse_params: Optional[Mapping[str, Any]] = None,
    ) -> None:
        # TODO: automatically retrieve the reverse_fn from the transform_fn
        super().__init__()
        self._transform_fn: TransformFn = transform_fn
        self.name: str = (
            name
            if name is not None
            else f"{transform_fn.__name__}"
            if hasattr(transform_fn, "__name__")
            else "transform"
        )
        self._reverse_fn: Optional[ReverseFn] = reverse_fn
        self._fit_params = {} if fit_params is None else fit_params
        self._transform_params = {} if transform_params is None else transform_params
        self._reverse_params = {} if reverse_params is None else reverse_params
        self._fitted: bool = False

    @property
    def invertible(self) -> bool:
        return self._reverse_fn is not None

    def forward(self, *args, **kwargs) -> TransformResult:
        return self.transform(*args, **kwargs)

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name}, reverse_fn={self.reverse_fn.__name__})"

    def get_state(self) -> Mapping[str, torch.Tensor]:
        return {k[len(_PREFIX) :]: v for k, v in self.state_dict().items() if k.startswith(_PREFIX)}

    @property
    def name(self) -> str:
        return self._transform_fn.__name__

    @property
    def reverse_fn(self) -> Optional[ReverseFn]:
        return self._reverse_fn

    def reverse(self, x: torch.Tensor, return_obj: bool = False) -> torch.Tensor:
        assert self.reverse_fn is not None, f"Reverse function not defined for {self.name}."
        assert self._fitted, "The transform must be fit first."
        return self.reverse_fn(x=x, **self._reverse_params, **self.get_state())

    def fit(self, x: torch.Tensor) -> None:
        transform_result: TransformResult = self._transform_fn(x=x, **self._fit_params)

        for key, value in transform_result.state.items():
            self.register_buffer(f"{_PREFIX}{key}", value)

        self._fitted: bool = True

    def transform(self, x: torch.Tensor, return_obj: bool = False) -> torch.Tensor:
        assert self._fitted, "The transform must be fit first."
        result = self._transform_fn(x=x, **self._transform_params, **self.get_state())
        if return_obj:
            return result
        else:
            return result.x
