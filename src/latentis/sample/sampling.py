from typing import Optional, Sequence, Union

import torch
from torch import nn

from latentis.space import LatentSpace, SpaceProperty
from latentis.types import Space


class Sampler(nn.Module):
    pass


class Uniform(Sampler):
    def __init__(self, random_seed: Optional[int] = None, suffix: Optional[str] = "_sampled"):
        """Subsample a latent space uniformly.

        Args:
            random_seed (Optional[int], optional): The random_seed for the random number generator. Defaults to None.
            suffix (Optional[str], optional): The suffix to append to the space name. Defaults to "_sampled".
        """
        super().__init__()

        self.generator = torch.Generator().manual_seed(random_seed) if random_seed is not None else None

        if suffix is None:
            suffix = ""
        self.suffix = suffix

    def forward(self, *spaces: Space, n: int) -> Union[Sequence[Space], Space]:
        """Samples n vectors uniformly at random from each space.

        Args:
            spaces (LatentSpace): The spaces to sample from.
            n (int): The number of vectors to sample from each space.

        Returns:
            Union[Sequence[LatentSpace], LatentSpace]: The sampled spaces.
        """
        assert len(spaces) > 0, "At least one space must be provided"
        assert len(set(len(space) for space in spaces)) == 1, "All spaces must have the same number of samples"
        assert n > 0, f"n must be greater than 0, but is {n}"
        assert n <= len(spaces[0]), f"n must be smaller than the number of vectors, but is {n}"
        assert len(set(type(space) for space in spaces)) == 1, "All spaces must be of the same class"

        ids = torch.randperm(len(spaces[0]), generator=self.generator)[:n]

        if isinstance(spaces[0], LatentSpace):
            out = tuple(
                LatentSpace(
                    vectors=space.vectors[ids],
                    name=f"{space.name}{self.suffix}",
                    features={
                        SpaceProperty.SAMPLING_IDS: ids,
                        **{key: values[ids] for key, values in space.features.items()},
                    },
                )
                for space in spaces
            )
        else:
            out = tuple(space[ids] for space in spaces)

        return out[0] if len(out) == 1 else out
