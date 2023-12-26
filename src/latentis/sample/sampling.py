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


# # def fps_sampling(x: torch.Tensor, num_anchors: int, seed: int) -> LatentSpace:
# #     anchor_fps = F.normalize(x, p=2, dim=-1)
# #     anchor_fps = fps(anchor_fps, random_start=True, ratio=num_anchors / x.size(0))

# #     return {"keys": anchor_fps.cpu(), "anchors": x[anchor_fps.to(x.device)]}


# class KMeansSampler(Sampler):
#     def forward(self, *spaces: Space, n: int) -> Union[Sequence[Space], Space]:
#         pass


# def kmeans_sampling(x: torch.Tensor, num_anchors: int, seed: int) -> LatentSpace:
#     vectors = F.normalize(x, p=2, dim=-1)
#     clustered = KMeans(n_clusters=num_anchors, random_state=seed).fit_predict(vectors.cpu().numpy())

#     all_targets = sorted(set(clustered))
#     cluster2embeddings = {target: vectors[clustered == target] for target in all_targets}
#     cluster2centroid = {cluster: centroid.mean(dim=0).cpu().numpy() for cluster, centroid in cluster2embeddings.items()}
#     centroids = np.array(list(cluster2centroid.values()), dtype="float32")

#     index: FaissIndex = FaissIndex(d=vectors.shape[1])
#     index.add_vectors(list(zip(range(vectors.size(0)), vectors.cpu().numpy())), normalize=False)
#     centroids = index.search_by_vectors(query_vectors=centroids, k_most_similar=1, normalize=True)

#     ids = torch.as_tensor(
#         [list(sample2score.keys())[0] for sample2score in centroids], dtype=torch.long, device=x.device
#     )

#     return {"keys": ids.cpu(), "anchors": x[ids]}


# class AnchorSampler:
#     def __init__(self, name: str, sampling_func: Callable[[torch.Tensor], LatentSpace]) -> None:
#         self._name: str = name
#         self.sampling_func: Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]] = sampling_func

#     @property
#     def name(self) -> str:
#         return self._name

#     def sample(self, x: torch.Tensor, num_anchors: int, seed: int) -> LatentSpace:
#         return self.sampling_func(x=x, num_anchors=num_anchors, seed=seed)
