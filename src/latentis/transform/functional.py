import inspect
from dataclasses import dataclass, field
from typing import Callable, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F
from scipy.stats import ortho_group


def _handle_zeros(scale: torch.Tensor, copy=True, constant_mask=None):
    """Set scales of near constant features to 1.

    The goal is to avoid division by very small or zero values.

    Near constant features are detected automatically by identifying
    scales close to machine precision unless they are precomputed by
    the caller and passed with the `constant_mask` kwarg.

    Typically for standard scaling, the scales are the standard
    deviation while near constant features are better detected on the
    computed variances which are closer to machine precision by
    construction.
    """
    eps = torch.finfo(scale.dtype).eps

    # if we are fitting on 1D tensors, scale might be a scalar
    if scale.ndim == 0:
        return 1 if scale == 0 else scale
    elif isinstance(scale, torch.Tensor):
        if constant_mask is None:
            # Detect near constant values to avoid dividing by a very small
            # value that could lead to surprising results and numerical
            # stability issues.
            constant_mask = scale < 10 * eps

        if copy:
            # New tensor to avoid side-effects
            scale = scale.clone()
        scale[constant_mask] = 1.0
        scale[scale == 0.0] = 1.0
        return scale


@dataclass
class TransformResult:
    x: torch.Tensor
    state: Mapping[str, torch.Tensor] = field(default_factory=dict)
    _state_key = "state"
    _x_key = "x"

    def __repr__(self) -> str:
        return f"TransformResult({self._x_key}={self.x}, {self._state_key}={self.state})"

    def as_dict(self) -> Mapping[str, torch.Tensor]:
        return {self._x_key: self.x, **self.state}


TransformFn = Callable[..., TransformResult]
ReverseFn = Callable[..., torch.Tensor]


# TODO: add LatentSpace support? (Space)
def transform_fn(
    name: Optional[str] = None, reverse_fn: Optional[ReverseFn] = None, state: Optional[Sequence[str]] = None
):
    # TODO: use name to register the transform_fn

    def decorator(transform_fn: TransformFn):
        transform_name = transform_fn.__name__ if name is None else name

        transform_fn_args = inspect.getfullargspec(transform_fn).args
        assert transform_fn_args == [
            "x"
        ], f"transform_fn should only have 'x' as a positional argument. transform_fn: {transform_fn_args}"

        if state is not None:
            assert isinstance(state, Sequence), f"state must be a sequence. state: {state}"
            assert all(isinstance(s, str) for s in state), f"state must be a sequence of strings. state: {state}"

            set_state = set(state)
        else:
            set_state = set()

        transform_fn_kwonlyargs = inspect.getfullargspec(transform_fn).kwonlyargs
        assert (
            set.intersection(set_state, set(transform_fn_kwonlyargs)) == set_state
        ), f"state mismatch while registering {transform_name}. transform_fn: {transform_fn_kwonlyargs}, state: {set_state}"

        # check that the reverse_fn has a compatible signature with the transform_fn
        if reverse_fn is not None:
            # TODO: are we sure about this check?
            assert len(set_state) > 0, "state must be specified when reverse_fn is specified."

            reverse_fn_args = inspect.getfullargspec(reverse_fn).args
            assert reverse_fn_args == [
                "x"
            ], f"Error registering {transform_name}: reverse_fn should only have 'x' as a positional argument. reverse_fn: {reverse_fn_args}"

            reverse_fn_kwonlyargs = inspect.getfullargspec(reverse_fn).kwonlyargs

            assert set_state == set(transform_fn_kwonlyargs).intersection(set(reverse_fn_kwonlyargs)), (
                f"state mismatch while registering {transform_name} with its reverse {reverse_fn.__name__}. "
                f"transform_fn: {transform_fn_kwonlyargs}, reverse_fn: {reverse_fn_kwonlyargs}. "
                f"state: {set_state}"
            )

        transform_fn._reverse_fn = reverse_fn

        return transform_fn

    return decorator


def centering_reverse(x: torch.Tensor, *, shift: torch.Tensor) -> torch.Tensor:
    return x + shift


@transform_fn(name="centering", reverse_fn=centering_reverse, state=["shift"])
def centering(x: torch.Tensor, *, shift: Optional[torch.Tensor] = None) -> TransformResult:
    shift = x.mean(dim=0) if shift is None else shift
    x = x - shift

    state = {"shift": shift}

    return TransformResult(x=x, state=state)


def std_scaling_reverse(x: torch.Tensor, *, scale: torch.Tensor) -> torch.Tensor:
    return x * scale


@transform_fn(name="std_scaling", reverse_fn=std_scaling_reverse, state=["scale"])
def std_scaling(x: torch.Tensor, *, scale: Optional[torch.Tensor] = None) -> TransformResult:
    if scale is None:
        scale = x.std(dim=0)
        scale[scale == 0.0] = 1.0

    x = x / scale
    state = {"scale": scale}

    return TransformResult(x=x, state=state)


def standard_scaling_reverse(x: torch.Tensor, *, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return (x * scale) + shift


@transform_fn(name="standard_scaling", reverse_fn=standard_scaling_reverse, state=["shift", "scale"])
def standard_scaling(
    x: torch.Tensor, *, shift: Optional[torch.Tensor] = None, scale: Optional[torch.Tensor] = None
) -> TransformResult:
    centering_result = centering(x=x, shift=shift)
    std_scaling_result = std_scaling(x=centering_result.x, scale=scale)

    state = centering_result.state
    state.update(std_scaling_result.state)

    return TransformResult(x=std_scaling_result.x, state=state)


@transform_fn(name="lp_norm")
def lp_normalize(x: torch.Tensor, *, p: int) -> TransformResult:
    x = F.normalize(x, p=p, dim=1)

    return TransformResult(x=x)


@transform_fn(name="l2_norm")
def l2_normalize(x: torch.Tensor) -> TransformResult:
    return lp_normalize(x=x, p=2)


def isotropic_scaling_reverse(x: torch.Tensor, *, scale: torch.Tensor) -> torch.Tensor:
    return x / scale


@transform_fn(name="isotropic_scaling", reverse_fn=isotropic_scaling_reverse, state=["scale"])
def isotropic_scaling(x: torch.Tensor, *, scale: torch.Tensor) -> TransformResult:
    x = x * scale

    state = {"scale": scale}

    return TransformResult(x=x, state=state)


@transform_fn(name="random_isotropic_scaling", reverse_fn=isotropic_scaling_reverse, state=["scale"])
def random_isotropic_scaling(
    x: torch.Tensor,
    *,
    low: Optional[float] = None,
    high: Optional[float] = None,
    random_seed: Optional[int] = None,
    scale: Optional[torch.Tensor] = None,
) -> TransformResult:
    assert (low is None and high is None) or low < high, f"low must be lower than high. low: {low}, high: {high}"
    assert (
        sum((scale is not None, (low is not None and high is not None and random_seed is not None))) == 1
    ), f"Either scale or (low, high, and random_seed) must be specified. scale: {scale}, low: {low}, high: {high}, random_seed: {random_seed}"

    scale = (
        (
            torch.rand(size=(1,), dtype=x.dtype, generator=torch.Generator().manual_seed(random_seed)) * (high - low)
            + low
        )
        if scale is None
        else scale
    )

    return isotropic_scaling(x=x, scale=scale)


def dimension_permutation_reverse(x: torch.Tensor, *, permutation: torch.Tensor) -> torch.Tensor:
    inverse_permutation = torch.zeros_like(permutation, dtype=torch.long, device=permutation.device)
    inverse_permutation[permutation] = torch.arange(len(permutation), dtype=torch.long, device=permutation.device)
    return x[:, inverse_permutation]


@transform_fn(name="dimension_permutation", reverse_fn=dimension_permutation_reverse, state=["permutation"])
def dimension_permutation(x: torch.Tensor, *, permutation: torch.Tensor) -> TransformResult:
    return TransformResult(x=x[:, permutation], state={"permutation": permutation})


@transform_fn(name="random_dimension_permutation", reverse_fn=dimension_permutation_reverse, state=["permutation"])
def random_dimension_permutation(
    x: torch.Tensor, *, random_seed: Optional[int] = None, permutation: Optional[torch.Tensor] = None
) -> TransformResult:
    assert (
        sum((permutation is not None, (random_seed is not None))) == 1
    ), f"Either permutation or random_seed must be specified. permutation: {permutation}, random_seed: {random_seed}"

    permutation = (
        torch.as_tensor(
            torch.randperm(x.shape[1], generator=torch.Generator().manual_seed(random_seed)), dtype=torch.long
        )
        if permutation is None
        else permutation
    )

    return dimension_permutation(x=x, permutation=permutation)


def isometry_reverse(x: torch.Tensor, *, matrix: torch.Tensor) -> torch.Tensor:
    return x @ matrix.T


@transform_fn(name="isometry", reverse_fn=isometry_reverse, state=["matrix"])
def isometry(x: torch.Tensor, *, matrix: torch.Tensor) -> TransformResult:
    return TransformResult(x=x @ matrix, state={"matrix": matrix})


@transform_fn(name="random_isometry", reverse_fn=isometry_reverse, state=["matrix"])
def random_isometry(
    x: torch.Tensor, *, random_seed: Optional[int] = None, matrix: Optional[torch.Tensor] = None
) -> TransformResult:
    assert (
        sum((matrix is not None, (random_seed is not None))) == 1
    ), f"Either isometry or random_seed must be specified. isometry: {matrix}, random_seed: {random_seed}"

    matrix = (
        torch.as_tensor(ortho_group.rvs(x.shape[1], random_state=random_seed), dtype=x.dtype)
        if matrix is None
        else matrix
    )

    return isometry(x=x, matrix=matrix)
