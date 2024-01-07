from typing import Any, Mapping

import pytest
import torch

import latentis.transform.functional as FL
from latentis.transform import Transform


@pytest.mark.parametrize(
    "transform_fn,fit_params",
    [
        (FL.centering, {}),
        (FL.l2_normalize, {}),
        (FL.std_scaling, {}),
        (FL.standard_scaling, {}),
        (FL.random_isotropic_scaling, dict(low=0.5, high=2.0, random_seed=42)),
        (FL.isotropic_scaling, dict(scale=torch.tensor([1.0]))),
        (FL.random_dimension_permutation, dict(random_seed=42)),
        (FL.random_isometry, dict(random_seed=42)),
    ],
)
def test_functional_transforms(
    transform_fn: FL.TransformFn,
    fit_params: Mapping[str, Any],
    tensor_space_with_ref,
):
    space, reference = tensor_space_with_ref

    space_out1: FL.TransformResult = transform_fn(x=space, **fit_params)
    transform = Transform(transform_fn=transform_fn, reverse_fn=transform_fn._reverse_fn, fit_params=fit_params)
    transform.fit(x=space)
    space_out2 = transform.transform(x=space, return_obj=True)
    assert torch.allclose(space_out1.x, space_out2.x)

    if transform.invertible:
        rev_out1 = transform.reverse(x=space_out1.x)
        rev_out2 = transform_fn._reverse_fn(x=space_out1.x, **space_out1.state)

        assert torch.allclose(rev_out1, rev_out2)
        assert torch.allclose(space, rev_out1)
