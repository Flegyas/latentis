import functools

import pytest
import torch

from latentis.transform import (
    L2,
    Centering,
    IsotropicScaling,
    RandomDimensionPermutation,
    RandomIsometry,
    RandomIsotropicScaling,
    StandardScaling,
    STDScaling,
    Transform,
)


@pytest.mark.parametrize(
    "transform_type,invertible",
    [
        (Centering, True),
        (L2, False),
        (STDScaling, True),
        (StandardScaling, True),
        (functools.partial(RandomIsotropicScaling, low=0.5, high=2.0, random_seed=42), True),
        (functools.partial(IsotropicScaling, scale=1.0), True),
        (functools.partial(RandomDimensionPermutation, random_seed=42), True),
        (functools.partial(RandomIsometry, random_seed=42), True),
    ],
)
def test_functional_transforms(
    transform_type: Transform,
    invertible: bool,
    tensor_space_with_ref,
):
    space, reference = tensor_space_with_ref

    transform = transform_type()
    transform.fit(reference=reference)

    out1 = transform(x=space)
    out2 = transform(x=space, reference=reference)
    out3 = transform_type()(x=space, reference=reference)

    assert torch.allclose(out1, out2)
    assert torch.allclose(out1, out3)

    rev_out1 = transform.reverse(x=out1)
    rev_out2 = transform.reverse(x=out2, reference=reference)
    rev_out3 = transform_type().reverse(x=out3, reference=reference)

    assert torch.allclose(rev_out1, rev_out2)
    assert torch.allclose(rev_out1, rev_out3)

    assert not invertible or torch.allclose(space, rev_out1)
