from typing import Any, Mapping, Optional

import torch
from torch import nn

from latentis.transform.functional import ReverseFn, TransformFn, TransformResult


class Transform(nn.Module):
    _PREFIX: str = "latentis_stat_"

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
        self._reverse_fn: Optional[ReverseFn] = reverse_fn
        self._fit_params = {} if fit_params is None else fit_params
        self._transform_params = {} if transform_params is None else transform_params
        self._reverse_params = {} if reverse_params is None else reverse_params
        self._fitted: bool = False
        self._name: Optional[str] = name

    @property
    def name(self) -> str:
        return (
            self._name
            if self._name is not None
            else f"{self._transform_fn.__name__}"
            if hasattr(self._transform_fn, "__name__")
            else "transform"
        )

    @property
    def invertible(self) -> bool:
        return self._reverse_fn is not None

    def forward(self, *args, **kwargs) -> TransformResult:
        return self.transform(*args, **kwargs)

    def __repr__(self):
        return f"name={self.name}, reverse_fn={self.reverse_fn.__name__ if self.reverse_fn is not None else None})"

    def get_state(self) -> Mapping[str, torch.Tensor]:
        return {k[len(Transform._PREFIX) :]: v for k, v in self.state_dict().items() if k.startswith(Transform._PREFIX)}

    @property
    def reverse_fn(self) -> Optional[ReverseFn]:
        return self._reverse_fn

    def reverse(self, x: torch.Tensor, return_obj: bool = False) -> torch.Tensor:
        assert self.reverse_fn is not None, f"Reverse function not defined for {self.name}."
        assert self._fitted, "The transform must be fit first."
        return self.reverse_fn(x=x, **self._reverse_params, **self.get_state())

    def fit(self, x: torch.Tensor, y=None) -> None:
        transform_result: TransformResult = self._transform_fn(x=x, **self._fit_params)

        for key, value in transform_result.state.items():
            self.register_buffer(f"{Transform._PREFIX}{key}", value)

        self._fitted: bool = True

        return self

    def transform(self, x: torch.Tensor, return_obj: bool = False) -> torch.Tensor:
        assert self._fitted, "The transform must be fit first."
        result = self._transform_fn(x=x, **self._transform_params, **self.get_state())
        if return_obj:
            return result
        else:
            return result.x

    def set_params(self, **params) -> "Transform":
        for key, value in params.items():
            if key == "transform_fn":
                self._transform_fn = value
            elif key == "reverse_fn":
                self._reverse_fn = value
            elif key in self._fit_params:
                self._fit_params[key] = value
            elif key in self._transform_params:
                self._transform_params[key] = value
            elif key in self._reverse_params:
                self._reverse_params[key] = value
            else:
                raise ValueError(f"Parameter {key} not found.")
        return self
