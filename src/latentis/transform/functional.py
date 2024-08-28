from typing import Callable, Mapping, Optional

import torch
import torch.nn.functional as F
from scipy.stats import ortho_group


def _handle_zeros(x: torch.Tensor, copy=True, constant_mask=None):
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
    eps = torch.finfo(x.dtype).eps

    # if we are fitting on 1D tensors, scale might be a scalar
    if x.ndim == 0:
        return 1 if x == 0 else x
    elif isinstance(x, torch.Tensor):
        if constant_mask is None:
            # Detect near constant values to avoid dividing by a very small
            # value that could lead to surprising results and numerical
            # stability issues.
            constant_mask = x < 10 * eps

        if copy:
            # New tensor to avoid side-effects
            x = x.clone()
        x[constant_mask] = 1.0
        x[x == 0.0] = 1.0
        return x


TransformFn = Callable[..., torch.Tensor]
InverseFn = Callable[..., torch.Tensor]
State = Mapping[str, torch.Tensor]
StateFn = Callable[..., State]

# TODO: add Space support? (Space)
# def transform_fn(
#     name: Optional[str] = None, inverse_fn: Optional[inverseFn] = None, state: Optional[Sequence[str]] = None
# ):
#     # TODO: use name to register the transform_fn

#     def decorator(transform_fn: TransformFn):
#         transform_name = transform_fn.__name__ if name is None else name

#         transform_fn_args = inspect.getfullargspec(transform_fn).args
#         assert transform_fn_args == [
#             "x"
#         ], f"transform_fn should only have 'x' as a positional argument. transform_fn: {transform_fn_args}"

#         if state is not None:
#             assert isinstance(state, Sequence), f"state must be a sequence. state: {state}"
#             assert all(isinstance(s, str) for s in state), f"state must be a sequence of strings. state: {state}"

#             set_state = set(state)
#         else:
#             set_state = set()

#         transform_fn_kwonlyargs = inspect.getfullargspec(transform_fn).kwonlyargs
#         assert (
#             set.intersection(set_state, set(transform_fn_kwonlyargs)) == set_state
#         ), f"state mismatch while registering {transform_name}. transform_fn: {transform_fn_kwonlyargs}, state: {set_state}"

#         # check that the inverse_fn has a compatible signature with the transform_fn
#         if inverse_fn is not None:
#             # TODO: are we sure about this check?
#             assert len(set_state) > 0, "state must be specified when inverse_fn is specified."

#             inverse_fn_args = inspect.getfullargspec(inverse_fn).args
#             assert inverse_fn_args == [
#                 "x"
#             ], f"Error registering {transform_name}: inverse_fn should only have 'x' as a positional argument. inverse_fn: {inverse_fn_args}"

#             inverse_fn_kwonlyargs = inspect.getfullargspec(inverse_fn).kwonlyargs

#             assert set_state == set(transform_fn_kwonlyargs).intersection(set(inverse_fn_kwonlyargs)), (
#                 f"state mismatch while registering {transform_name} with its inverse {inverse_fn.__name__}. "
#                 f"transform_fn: {transform_fn_kwonlyargs}, inverse_fn: {inverse_fn_kwonlyargs}. "
#                 f"state: {set_state}"
#             )

#         transform_fn._inverse_fn = inverse_fn

#         return transform_fn

#     return decorator


def centering_transform(x: torch.Tensor, *, shift: torch.Tensor) -> torch.Tensor:
    return x - shift


def centering_inverse(x: torch.Tensor, *, shift: torch.Tensor) -> torch.Tensor:
    return x + shift


def centering_state(x: torch.Tensor) -> State:
    return {"shift": x.mean(dim=0)}


def std_scaling_transform(x: torch.Tensor, *, scale: torch.Tensor) -> torch.Tensor:
    return x / scale


def std_scaling_inverse(x: torch.Tensor, *, scale: torch.Tensor) -> torch.Tensor:
    return x * scale


def std_scaling_state(x: torch.Tensor) -> State:
    scale = x.std(dim=0)
    scale = _handle_zeros(scale)
    return {"scale": scale}


def standard_scaling_transform(
    x: torch.Tensor, *, shift: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    x = centering_transform(x=x, shift=shift)
    x = std_scaling_transform(x=x, scale=scale)
    return x


def standard_scaling_inverse(
    x: torch.Tensor, *, shift: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    return (x * scale) + shift


def standard_scaling_state(x: torch.Tensor) -> State:
    return {**centering_state(x=x), **std_scaling_state(x=x)}


def lp_normalize_transform(x: torch.Tensor, *, p: int) -> torch.Tensor:
    return F.normalize(x, p=p, dim=1)


def l2_normalize_transform(x: torch.Tensor) -> torch.Tensor:
    return lp_normalize_transform(x=x, p=2)


def isotropic_scaling_transform(
    x: torch.Tensor, *, scale: torch.Tensor
) -> torch.Tensor:
    return x * scale


def isotropic_scaling_inverse(x: torch.Tensor, *, scale: torch.Tensor) -> torch.Tensor:
    return x / scale


def dimension_permutation_transform(
    x: torch.Tensor, *, permutation: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    return x.index_select(dim=dim, index=permutation)


def dimension_permutation_inverse(
    x: torch.Tensor, *, permutation: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    inverse_permutation = torch.zeros_like(
        permutation, dtype=torch.long, device=permutation.device
    )
    inverse_permutation[permutation] = torch.arange(
        len(permutation), dtype=torch.long, device=permutation.device
    )
    return x.index_select(dim=dim, index=inverse_permutation)


def random_dimension_permutation_state(x: torch.Tensor, random_seed: int) -> State:
    return {
        "permutation": torch.as_tensor(
            torch.randperm(
                x.shape[1], generator=torch.Generator().manual_seed(random_seed)
            ),
            dtype=torch.long,
            device=x.device,
        )
    }


def isometry_transform(x: torch.Tensor, *, matrix: torch.Tensor) -> torch.Tensor:
    return x @ matrix


def isometry_inverse(x: torch.Tensor, *, matrix: torch.Tensor) -> torch.Tensor:
    return x @ matrix.T


def random_isometry_state(
    x: torch.Tensor, *, random_seed: Optional[int] = None
) -> State:
    matrix = torch.as_tensor(
        ortho_group.rvs(x.shape[1], random_state=random_seed),
        dtype=x.dtype,
        device=x.device,
    )
    return {"matrix": matrix}
