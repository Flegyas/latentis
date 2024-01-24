from typing import Any, Mapping

import pytest
import torch

import latentis.transform.functional as FL
from latentis.transform._abstract import SimpleTransform, Transform
from latentis.transform.base import (
    Centering,
    InverseTransform,
    IsotropicScaling,
    LPNorm,
    RandomDimensionPermutation,
    RandomIsometry,
    StandardScaling,
    STDScaling,
)


@pytest.mark.parametrize(
    "transform_fn,inverse_fn,state_fn,transform,fit_params,transform_params,inverse_params",
    [
        (FL.centering_transform, FL.centering_inverse, FL.centering_state, Centering, None, None, None),
        (FL.l2_normalize_transform, None, None, lambda: LPNorm(p=2), None, None, None),
        (FL.lp_normalize_transform, None, None, lambda: LPNorm(p=3), None, dict(p=3), None),
        (FL.std_scaling_transform, FL.std_scaling_inverse, FL.std_scaling_state, STDScaling, None, None, None),
        (
            FL.standard_scaling_transform,
            FL.standard_scaling_inverse,
            FL.standard_scaling_state,
            StandardScaling,
            None,
            None,
            None,
        ),
        (
            FL.isotropic_scaling_transform,
            FL.isotropic_scaling_inverse,
            None,
            lambda: IsotropicScaling(scale=2.0),
            dict(scale=2.0),
            dict(scale=2.0),
            dict(scale=2.0),
        ),
        (
            FL.dimension_permutation_transform,
            FL.dimension_permutation_inverse,
            FL.random_dimension_permutation_state,
            lambda: RandomDimensionPermutation(random_seed=42),
            dict(random_seed=42),
            None,
            None,
        ),
        (
            FL.isometry_transform,
            FL.isometry_inverse,
            FL.random_isometry_state,
            lambda: RandomIsometry(random_seed=42),
            dict(random_seed=42),
            None,
            None,
        ),
    ],
)
def test_functional_transforms(
    transform_fn: FL.TransformFn,
    tensor_space_with_ref,
    transform: Transform,
    state_fn: FL.StateFn,
    inverse_fn: FL.InverseFn,
    fit_params: Mapping[str, Any],
    transform_params: Mapping[str, Any],
    inverse_params: Mapping[str, Any],
):
    assert transform is not None
    fit_params = fit_params or {}
    transform_params = transform_params or {}
    inverse_params = inverse_params or {}

    space, reference = tensor_space_with_ref
    state = state_fn(x=space, **fit_params) if state_fn else {}
    space_out1: FL.TransformResult = transform_fn(x=space, **state, **transform_params)

    simple_transform = SimpleTransform(
        transform_fn=transform_fn,
        inverse_fn=inverse_fn,
        state_fn=state_fn,
        state_params=fit_params,
        transform_params=transform_params,
        inverse_params=inverse_params,
    )
    simple_transform.fit(x=space)
    space_out2, _ = simple_transform.transform(x=space)

    transform = transform().fit(x=space)
    space_out3, _ = transform.transform(x=space)

    assert torch.allclose(space_out1, space_out2)
    assert torch.allclose(space_out1, space_out3)

    if simple_transform.invertible:
        rev_out1, _ = simple_transform.inverse_transform(x=space_out1)
        rev_out2 = inverse_fn(x=space_out1, **inverse_params, **state)
        rev_out3, _ = transform.inverse_transform(x=space_out1)
        inverse_transform = InverseTransform(transform=transform)

        rev_out4, _ = inverse_transform.transform(x=space_out1, y=None)

        assert torch.allclose(space, rev_out1)
        assert torch.allclose(rev_out1, rev_out2)
        assert torch.allclose(rev_out2, rev_out3)
        assert torch.allclose(rev_out3, rev_out4)
