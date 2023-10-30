from typing import Union

import pytest as pytest
import torch
from torch import Tensor

from latentis.sampling import Uniform
from latentis.space import LatentSpace, SpaceProperty


def test_uniform_sampler(space1: Union[LatentSpace, Tensor], space2: Union[LatentSpace, Tensor]):
    # Test invalid inputs
    uniform = Uniform()
    with pytest.raises(AssertionError):
        _ = uniform(space1, n=-1)

    with pytest.raises(AssertionError):
        _ = uniform(space1, n=len(space1) + 1)

    with pytest.raises(AssertionError):
        _, _ = uniform(space1, space2, n=10)

    # Test suffix
    uniform = Uniform(random_seed=0, suffix="_custom")
    subspace1 = uniform(space1, n=10)
    if isinstance(space1, LatentSpace):
        assert subspace1.name.startswith(space1.name)
        assert subspace1.name.endswith("_custom")
        assert subspace1.name == space1.name + "_custom"

    # Test sampling size
    uniform = Uniform(random_seed=0)
    subspace2 = uniform(space2, n=10)
    assert len(subspace1) == 10 == len(subspace2)

    # Test sampling ids are present but different
    if isinstance(space1, LatentSpace) and isinstance(space2, LatentSpace):
        assert not torch.all(
            subspace1.properties[SpaceProperty.SAMPLING_IDS] == subspace2.properties[SpaceProperty.SAMPLING_IDS]
        )

    # Parallel sampling given same seed
    uniform = Uniform(random_seed=0)
    subspace1 = uniform(space1, n=5)
    uniform = Uniform(random_seed=0)
    space1_2 = uniform(space1, n=5)
    assert len(subspace1) == 5 == len(space1_2)

    # Test sampling ids are present and the same
    if isinstance(space1, LatentSpace) and isinstance(space2, LatentSpace):
        assert torch.all(
            subspace1.properties[SpaceProperty.SAMPLING_IDS] == space1_2.properties[SpaceProperty.SAMPLING_IDS]
        ).item()

    # Parallel sampling in the same call
    uniform = Uniform()
    subspace1, space1_2 = uniform(space1, space1, n=10)
    assert len(subspace1) == 10 == len(space1_2)

    # Test sampling ids are present and the same
    if isinstance(space1, LatentSpace) and isinstance(space2, LatentSpace):
        assert torch.all(
            subspace1.properties[SpaceProperty.SAMPLING_IDS] == space1_2.properties[SpaceProperty.SAMPLING_IDS]
        )
