import functools

import pytest
import torch

from latentis.brick import (
    L2,
    Brick,
    Centering,
    IsotropicScaling,
    RandomDimensionPermutation,
    RandomIsometry,
    RandomIsotropicScaling,
    StandardScaling,
    STDScaling,
)
from latentis.bridge import BrickSequence
from latentis.space import LatentSpace


@pytest.mark.parametrize(
    "bricks_type, invertible",
    [
        ((Centering,), True),
        ((STDScaling,), True),
        ((StandardScaling,), True),
        ((L2,), False),
        (
            (
                Centering,
                STDScaling,
            ),
            True,
        ),
        ((STDScaling, Centering), True),
        ((L2, Centering), False),
        ((L2, Centering, L2), False),
        ((functools.partial(IsotropicScaling, scale=10.0),), True),
        ((functools.partial(RandomIsotropicScaling, low=0.5, high=2.0, random_seed=0),), True),
        ((functools.partial(RandomDimensionPermutation, random_seed=0),), True),
        ((functools.partial(RandomIsometry, random_seed=0),), True),
    ],
)
def test_brick_sequence(bricks_type: Brick, invertible: bool, space1):
    if isinstance(space1, LatentSpace):
        pytest.skip("Bricks only support tensors (for now).")

    manual_x = space1.clone()
    transforms = []
    for brick in bricks_type:
        transform = brick()
        transform.fit(manual_x)
        manual_x = transform(manual_x)
        transforms.append(transform)

    brick = BrickSequence(bricks=[brick() for brick in bricks_type])
    brick.fit(space1)
    auto_x = brick.forward(space1)

    assert torch.allclose(manual_x, auto_x)

    if invertible:
        for t in reversed(transforms):
            manual_x = t.reverse(manual_x)
        assert torch.allclose(manual_x, space1)

        auto_reverse = brick.reverse(auto_x)
        assert torch.allclose(auto_reverse, space1)
