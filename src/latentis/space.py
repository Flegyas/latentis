from __future__ import annotations

from enum import auto
from typing import TYPE_CHECKING, Any, Dict, Mapping, Optional, Sequence

if TYPE_CHECKING:
    from latentis.sampling import Sampler

import torch
from torch.utils.data import Dataset as TorchDataset

try:
    # be ready for 3.10 when it drops
    from enum import StrEnum
except ImportError:
    from backports.strenum import StrEnum


class SpaceProperty(StrEnum):
    SAMPLING_IDS = auto()


class LatentSpace(TorchDataset):
    def __init__(
        self,
        vectors: torch.Tensor,
        name: str,
        properties: Optional[Dict[str, Sequence[Any]]] = None,
    ):
        assert properties is None or all(len(values) == vectors.size(0) for values in properties.values())
        if properties is None:
            properties = {}
        self.vectors: torch.Tensor = vectors
        self.name: str = name
        self.properties = properties

    @staticmethod
    def like(
        space: LatentSpace,
        name: Optional[str],
        vectors: Optional[torch.Tensor] = None,
        properties: Optional[Dict[str, Sequence[Any]]] = None,
    ):
        """Create a new space with the arguments not provided taken from the given space.

        There is no copy of the vectors, so changes to the vectors of the new space will also affect the vectors of the given space.

        Args:
            space (LatentSpace): The space to copy.
            name (Optional[str]): The name of the new space.
            vectors (Optional[torch.Tensor], optional): The vectors of the new space.
            properties (Optional[Dict[str, Sequence[Any]]], optional): The properties of the new space.

        Returns:
            LatentSpace: The new space, with the arguments not provided taken from the given space.
        """
        if name is None:
            name = space.name
        if vectors is None:
            vectors = space.vectors
        if properties is None:
            properties = space.properties
        return LatentSpace(name=name, vectors=vectors, properties=properties)

    @property
    def shape(self) -> torch.Size:
        return self.vectors.shape

    def __getitem__(self, index: int) -> Mapping[str, torch.Tensor]:
        return {"x": self.vectors[index], **{key: values[index] for key, values in self.properties.items()}}

    def __len__(self) -> int:
        return self.vectors.size(0)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, vectors={self.vectors.shape}, properties={self.properties.keys()})"

    def sample(self, sampler: Sampler, n: int) -> "LatentSpace":
        """Sample n vectors from this space using the given sampler.

        Args:
            sampler (Sampler): The sampler to use.
            n (int): The number of vectors to sample.

        Returns:
            LatentSpace: The sampled space.

        """
        return sampler(self, n=n)

    # def translate(
    #     self,
    #     translation: LatentTranslator,
    # ):
    #     result = translation(x=self.vectors)

    #     return LatentSpace(
    #         name=self._name,
    #         vectors=result["target"],
    #         keys=self.key2index.keys(),
    #         labels=self.labels,
    #     )

    # @lru_cache
    # def to_faiss(self, normalize: bool, keys: Sequence[str]) -> FaissIndex:
    #     index: FaissIndex = FaissIndex(d=self.vectors.size(1))

    #     index.add_vectors(
    #         embeddings=list(zip(keys, self.vectors.cpu().numpy())),
    #         normalize=normalize,
    #     )

    #     return index

    # def to_relative(
    #     self,
    #     projection_name: str,
    #     projection_func: Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
    #     anchor_choice: str = None,
    #     seed: int = None,
    #     anchors: Optional[Mapping[str, torch.Tensor]] = None,
    #     num_anchors: int = None,
    # ) -> "RelativeSpace":
    #     assert self.encoding_type != "relative"  # TODO: for now
    #     assert (anchors is None) or (num_anchors is None)

    #     anchors = (
    #         self.get_anchors(anchor_choice=anchor_choice, seed=seed, num_anchors=num_anchors)
    #         if anchors is None
    #         else anchors
    #     )

    #     anchor_keys, anchor_latents = list(zip(*anchors.items()))
    #     anchor_latents = torch.stack(anchor_latents, dim=0).cpu()

    #     relative_vectors = projection_func(
    #         anchors=anchor_latents,
    #         points=self.vectors,
    #     )

    #     return RelativeSpace(
    #         keys=self.key2index.keys(),
    #         vectors=relative_vectors,
    #         labels=self.labels,
    #         encoder=self.encoder,
    #         anchors=anchor_keys,
    #         projection=projection_name,
    #         num_classes=self.num_classes,
    #     )


# class RelativeSpace(LatentSpace):
#     def __init__(
#         self,
#         keys: Sequence[str],
#         vectors: torch.Tensor,
#         labels: torch.Tensor,
#         anchors: Sequence[str],
#         projection: str,
#         num_classes: int,
#         encoder: str = None,
#     ):
#         super().__init__(
#             encoding_type="relative",
#             keys=keys,
#             num_classes=num_classes,
#             vectors=vectors,
#             labels=labels,
#             encoder=encoder,
#         )
#         self.anchors: Sequence[str] = anchors
#         self.projection: str = projection

#     def __repr__(self) -> str:
#         return f"LatentSpace(encoding_type={self.encoding_type}, projection={self.projection}, encoder={self.encoder}, shape={self.shape})"
