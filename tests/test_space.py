from typing import Union

from torch import Tensor

from latentis.sampling import Uniform
from latentis.space import LatentSpace


def test_space_len(space1: Union[LatentSpace, Tensor]):
    if isinstance(space1, LatentSpace):
        assert len(space1) == space1.vectors.shape[0]


def test_space_repr(space1: Union[LatentSpace, Tensor]):
    assert repr(space1)


def test_space_get(space1: Union[LatentSpace, Tensor]):
    item = space1[0]
    if isinstance(space1, LatentSpace):
        assert item["x"].shape[0] == space1.vectors.shape[-1]
        assert item["label"] == space1.properties["label"][0]


def test_space_sample_hook(space1: Union[LatentSpace, Tensor]):
    if isinstance(space1, LatentSpace):
        subspace = space1.sample(Uniform(), n=50)
        assert subspace.name == space1.name + "_sampled"
        assert subspace.properties["sampling_ids"].shape == (50,)
        assert len(subspace) == 50
