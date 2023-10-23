from abc import abstractmethod
from typing import Any, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F
from pytorch_lightning import seed_everything
from torch import nn

from latentis.transforms import Transform
from latentis.types import TransformType


class Translator(nn.Module):
    def __init__(self, name: str) -> None:
        super().__init__()
        self._name: str = name

    @property
    def name(self) -> str:
        return self._name

    @abstractmethod
    def fit(self, source_data: torch.Tensor, target_data: torch.Tensor) -> Mapping[str, Any]:
        raise NotImplementedError


class IdentityTranslator(Translator):
    def __init__(self) -> None:
        super().__init__("identity")

    def fit(self, *args, **kwargs) -> Mapping[str, Any]:
        return {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class AffineTranslator(Translator):
    def __init__(self) -> None:
        super().__init__("affine")

    def fit(self, source_data: torch.Tensor, target_data: torch.Tensor) -> Mapping[str, Any]:
        with torch.enable_grad():
            translation = nn.Linear(source_data.size(1), target_data.size(1), device=source_data.device)
            optimizer = torch.optim.Adam(translation.parameters(), lr=1e-3)

            for _ in range(100):
                optimizer.zero_grad()
                loss = F.mse_loss(translation(source_data), target_data)
                loss.backward()
                optimizer.step()
            self.translation = translation.cpu()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.translation(x)


class LSTSQTranslator(Translator):
    def __init__(self) -> None:
        super().__init__("lstsq")
        self.translation_matrix = None

    def fit(self, source_data: torch.Tensor, target_data: torch.Tensor) -> Mapping[str, Any]:
        translation_matrix = torch.linalg.lstsq(source_data, target_data).solution
        self.translation_matrix = torch.as_tensor(translation_matrix, dtype=torch.float32, device=source_data.device)

        return {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.translation_matrix


class LSTSQOrthoTranslator(Translator):
    def __init__(self) -> None:
        super().__init__("lstsq_ortho")

    def fit(self, source_data: torch.Tensor, target_data: torch.Tensor) -> Mapping[str, Any]:
        translation_matrix = torch.linalg.lstsq(source_data, target_data).solution
        U, _, Vt = torch.svd(translation_matrix)
        self.translation_matrix = U @ Vt.T

        return {}


class SVDTranslator(Translator):
    def __init__(self) -> None:
        super().__init__("svd")
        self.translation_matrix = None

    def fit(self, source_data: torch.Tensor, target_data: torch.Tensor) -> Mapping[str, Any]:
        # padding if necessary
        if source_data.size(1) < target_data.size(1):
            padded = torch.zeros_like(target_data)
            padded[:, : source_data.size(1)] = source_data
            source_data = padded
        elif source_data.size(1) > target_data.size(1):
            padded = torch.zeros_like(source_data)
            padded[:, : target_data.size(1)] = target_data
            target_data = padded

        #  Compute the translation vector that aligns A to B using SVD.
        assert source_data.size(1) == target_data.size(1)
        u, sigma, vt = torch.svd((target_data.T @ source_data).T)
        translation_matrix = u @ vt.T

        self.translation_matrix = torch.as_tensor(translation_matrix, dtype=torch.float32, device=source_data.device)
        sigma_rank = (~sigma.isclose(torch.zeros_like(sigma), atol=1e-1).bool()).sum().item()

        return {"sigma_rank": sigma_rank}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.translation_matrix


class LatentTranslation(nn.Module):
    def __init__(
        self,
        seed: int,
        translator: Translator,
        source_transforms: Optional[Sequence[TransformType]] = None,
        target_transforms: Optional[Sequence[TransformType]] = None,
    ) -> None:
        super().__init__()
        self.seed: int = seed
        self.translator: Translator = translator

        self.source_transforms: Sequence[Transform] = nn.ModuleList(
            source_transforms
            if isinstance(source_transforms, Sequence)
            else []
            if source_transforms is None
            else [source_transforms]
        )
        self.target_transforms: Sequence[Transform] = nn.ModuleList(
            target_transforms
            if isinstance(target_transforms, Sequence)
            else []
            if target_transforms is None
            else [target_transforms]
        )

    def fit(self, source_data: torch.Tensor, target_data: torch.Tensor) -> Mapping[str, Any]:
        seed_everything(self.seed)
        translator_info = self.translator.fit(source_data=source_data, target_data=target_data)

        # for transform in self.source_transforms:
        #     transform.fit(source_data=source_data)
        # for transform in self.target_transforms:
        #     transform.fit(target_data=target_data)
        self.register_buffer("source_data", source_data)
        self.register_buffer("target_data", target_data)

        return translator_info

    def forward(self, x: torch.Tensor, compute_info: bool = True) -> torch.Tensor:
        source_x = x
        for transform in self.source_transforms:
            source_x = transform(x=source_x, anchors=self.source_data)

        target_x = self.translator(source_x)

        for transform in reversed(self.target_transforms):
            target_x = transform.reverse(x=target_x, anchors=self.target_data)

        return {"source": source_x, "target": target_x, "info": {}}
