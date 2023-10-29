import pytest as pytest
import torch

from latentis.sampling import Uniform
from latentis.space import LatentSpace, SpaceProperty


def test_uniform_sampler(space1: LatentSpace, space2: LatentSpace):
    uniform = Uniform()
    with pytest.raises(AssertionError):
        _ = uniform(space1, n=-1)

    with pytest.raises(AssertionError):
        _ = uniform(space1, n=len(space1) + 1)

    with pytest.raises(AssertionError):
        _, _ = uniform(space1, space2, n=10)

    uniform = Uniform(random_seed=0, suffix="_custom")
    subspace1 = uniform(space1, n=10)
    assert subspace1.name.startswith(space1.name)
    assert subspace1.name.endswith("_custom")
    assert subspace1.name == space1.name + "_custom"

    uniform = Uniform(random_seed=0)
    subspace2 = uniform(space2, n=10)
    assert len(subspace1) == 10 == len(subspace2)
    assert not torch.all(
        subspace1.properties[SpaceProperty.SAMPLING_IDS] == subspace2.properties[SpaceProperty.SAMPLING_IDS]
    )

    uniform = Uniform(random_seed=0)
    subspace1 = uniform(space1, n=5)

    uniform = Uniform(random_seed=0)
    space1_2 = uniform(space1, n=5)
    assert len(subspace1) == 5 == len(space1_2)
    assert torch.all(
        subspace1.properties[SpaceProperty.SAMPLING_IDS] == space1_2.properties[SpaceProperty.SAMPLING_IDS]
    ).item()

    uniform = Uniform()
    subspace1, space1_2 = uniform(space1, space1, n=10)
    assert len(subspace1) == 10 == len(space1_2)
    assert torch.all(
        subspace1.properties[SpaceProperty.SAMPLING_IDS] == space1_2.properties[SpaceProperty.SAMPLING_IDS]
    )
