from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from latentis.pipeline.flow import Flow, NNPipeline
from latentis.space import Space
from latentis.transform import Identity, XTransformSequence
from latentis.transform.base import Centering, MeanLPNorm, StandardScaling, STDScaling
from latentis.transform.dim_matcher import ZeroPadding
from latentis.transform.translate import MatrixAligner
from latentis.transform.translate.aligner import SGDAffineAligner
from latentis.transform.translate.functional import lstsq_align_state, lstsq_ortho_align_state, svd_align_state
from latentis.utils import seed_everything

if TYPE_CHECKING:
    from latentis.types import LatentisSpace


def manual_svd_translation(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    # """Compute the translation vector that aligns A to B using SVD."""
    assert A.size(1) == B.size(1)
    u, s, vt = torch.svd((B.T @ A).T)
    R = u @ vt.T
    return R, s


class ManualLatentTranslation(nn.Module):
    def __init__(self, seed: int, centering: bool, std_correction: bool, l2_norm: bool, method: str) -> None:
        super().__init__()

        self.seed: int = seed
        self.centering: bool = centering
        self.std_correction: bool = std_correction
        self.l2_norm: bool = l2_norm
        self.method: str = method
        self.sigma_rank: Optional[float] = None

        self.translation_matrix: Optional[torch.Tensor]
        self.mean_encoding_anchors: Optional[torch.Tensor]
        self.mean_decoding_anchors: Optional[torch.Tensor]
        self.std_encoding_anchors: Optional[torch.Tensor]
        self.std_decoding_anchors: Optional[torch.Tensor]
        self.encoding_norm: Optional[torch.Tensor]
        self.decoding_norm: Optional[torch.Tensor]

    @torch.no_grad()
    def fit(self, x: torch.Tensor, y: torch.Tensor) -> None:
        if self.method == "absolute":
            return
        # First normalization: 0 centering
        if self.centering:
            mean_encoding_anchors: torch.Tensor = x.mean(dim=(0,))
            mean_decoding_anchors: torch.Tensor = y.mean(dim=(0,))
        else:
            mean_encoding_anchors: torch.Tensor = torch.as_tensor(0)
            mean_decoding_anchors: torch.Tensor = torch.as_tensor(0)

        if self.std_correction:
            std_encoding_anchors: torch.Tensor = x.std(dim=(0,))
            std_decoding_anchors: torch.Tensor = y.std(dim=(0,))
        else:
            std_encoding_anchors: torch.Tensor = torch.as_tensor(1)
            std_decoding_anchors: torch.Tensor = torch.as_tensor(1)

        self.encoding_dim: int = x.size(1)
        self.decoding_dim: int = y.size(1)

        self.register_buffer("mean_encoding_anchors", mean_encoding_anchors)
        self.register_buffer("mean_decoding_anchors", mean_decoding_anchors)
        self.register_buffer("std_encoding_anchors", std_encoding_anchors)
        self.register_buffer("std_decoding_anchors", std_decoding_anchors)

        x = (x - mean_encoding_anchors) / std_encoding_anchors
        y = (y - mean_decoding_anchors) / std_decoding_anchors

        self.register_buffer("encoding_norm", x.norm(p=2, dim=-1).mean())
        self.register_buffer("decoding_norm", y.norm(p=2, dim=-1).mean())

        # Second normalization: scaling
        if self.l2_norm:
            x = F.normalize(x, p=2, dim=-1)
            y = F.normalize(y, p=2, dim=-1)

        if self.method == "linear":
            with torch.random.fork_rng():
                seed_everything(seed=self.seed)
                with torch.enable_grad():
                    translation = nn.Linear(
                        x.size(1),
                        y.size(1),
                        device=x.device,
                        dtype=x.dtype,
                        bias=True,
                    )
                    optimizer = torch.optim.Adam(translation.parameters(), lr=1e-3)

                    for _ in range(20):
                        optimizer.zero_grad()
                        loss = F.mse_loss(translation(x), y)
                        loss.backward()
                        optimizer.step()
                    self.translation = translation.cpu()
            return

        if self.method == "svd":
            # padding if necessary
            if x.size(1) < y.size(1):
                padded = torch.zeros_like(y)
                padded[:, : x.size(1)] = x
                x = padded
            elif x.size(1) > y.size(1):
                padded = torch.zeros_like(x)
                padded[:, : y.size(1)] = y
                y = padded

                self.encoding_anchors = x
                self.decoding_anchors = y

            translation_matrix, sigma = manual_svd_translation(A=x, B=y)
            self.sigma_rank = (~sigma.isclose(torch.zeros_like(sigma), atol=1e-1).bool()).sum().item()
        elif self.method == "lstsq":
            translation_matrix = torch.linalg.lstsq(x, y).solution
        elif self.method == "lstsq+ortho":
            translation_matrix = torch.linalg.lstsq(x, y).solution
            U, _, Vt = torch.svd(translation_matrix)
            translation_matrix = U @ Vt.T
        else:
            raise NotImplementedError

        translation_matrix = torch.as_tensor(translation_matrix, dtype=x.dtype, device=x.device)
        self.register_buffer("translation_matrix", translation_matrix)

        self.translation = lambda x: x @ self.translation_matrix

        return self

    def transform(self, X: torch.Tensor, compute_info: bool = True) -> torch.Tensor:
        if self.method == "absolute":
            return {"source": X, "target": X, "info": {}}

        encoding_x = (X - self.mean_encoding_anchors) / self.std_encoding_anchors

        if self.l2_norm:
            encoding_x = F.normalize(encoding_x, p=2, dim=-1)

        if self.method == "svd" and self.encoding_dim < self.decoding_dim:
            padded = torch.zeros(X.size(0), self.decoding_dim, device=X.device, dtype=X.dtype)
            padded[:, : self.encoding_dim] = encoding_x
            encoding_x = padded

        decoding_x = self.translation(encoding_x)

        decoding_x = decoding_x[:, : self.decoding_dim]

        # restore scale
        if self.l2_norm:
            decoding_x = decoding_x * self.decoding_norm

        # restore center
        decoding_x = (decoding_x * self.std_decoding_anchors) + self.mean_decoding_anchors

        info = {}
        if compute_info:
            pass

        return {"source": encoding_x, "target": decoding_x, "info": info}


_RANDOM_SEED = 0


@pytest.mark.parametrize(
    "manual_method, aligner_factory",
    [
        ("svd", lambda: MatrixAligner("svd", align_fn_state=svd_align_state)),
        ("lstsq", lambda: MatrixAligner("lstsq", align_fn_state=lstsq_align_state)),
        ("lstsq+ortho", lambda: MatrixAligner("lstsq+ortho", align_fn_state=lstsq_ortho_align_state)),
        ("linear", lambda: SGDAffineAligner(num_steps=20, lr=1e-3, random_seed=0)),
    ],
)
@pytest.mark.parametrize(
    "manual_centering, manual_std_correction, manual_l2_norm, x_transform, y_transform",
    [
        (
            True,
            True,
            False,
            XTransformSequence([Centering(), STDScaling()]),
            XTransformSequence([Centering(), STDScaling()]),
        ),
        (
            True,
            True,
            False,
            StandardScaling(),
            StandardScaling(),
        ),
        (
            True,
            False,
            False,
            Centering(),
            Centering(),
        ),
        (
            True,
            False,
            True,
            XTransformSequence([Centering(), MeanLPNorm(p=2)]),
            XTransformSequence([Centering(), MeanLPNorm(p=2)]),
        ),
        (False, False, True, MeanLPNorm(p=2), MeanLPNorm(p=2)),
    ],
)
def test_manual_translation(
    parallel_spaces: Tuple[LatentisSpace, LatentisSpace],
    manual_method,
    aligner_factory,
    manual_centering,
    manual_std_correction,
    manual_l2_norm,
    x_transform,
    y_transform,
):
    manual_translator = ManualLatentTranslation(
        seed=_RANDOM_SEED,
        centering=manual_centering,
        std_correction=manual_std_correction,
        l2_norm=manual_l2_norm,
        method=manual_method,
    )
    # translator = Translator(
    #     # random_seed=0,
    #     aligner=aligner_factory(),
    #     x_transform=x_transform,
    #     y_transform=y_transform,
    # )
    translator = NNPipeline(
        name="translator",
        flows={
            "fit": Flow(name="fit", inputs=["fit_source", "fit_target"], outputs="estimator")
            .add(block="x_transform", method="fit_transform", inputs="fit_source:x", outputs="x")
            .add(block="y_transform", method="fit_transform", inputs="fit_target:x", outputs="y")
            .add(block="padder", method="fit_transform", inputs=["x", "y"], outputs=["padded_x", "padded_y"])
            .add(block="aligner", method="fit", inputs=["padded_x:x", "padded_y:y"], outputs=["aligner_state"]),
            #
            "transform": Flow(name="transform", inputs=["source"], outputs=["translated_x"])
            .add(block="x_transform", method="transform", inputs="source:x", outputs="x")
            .add(block="padder", method="transform", inputs="x", outputs="x")
            .add(block="aligner", method="transform", inputs="x", outputs="x")
            .add(block="padder", method="inverse_transform", inputs="x:y", outputs="x")
            .add(block="y_transform", method="inverse_transform", inputs="x", outputs="translated_x"),
        },
        blocks={
            "x_transform": x_transform,
            "y_transform": y_transform,
            "aligner": aligner_factory(),
            "padder": ZeroPadding() if manual_method == "svd" else Identity(),
        },
    )

    A, B = parallel_spaces
    A = A.vectors if isinstance(A, Space) else A
    B = B.vectors if isinstance(B, Space) else B
    manual_translator.fit(A, B)
    translator.run(flow="fit", fit_source=A, fit_target=B)

    manual_output = manual_translator.transform(A)["target"]
    latentis_output = translator.run(flow="transform", source=A)

    assert torch.allclose(
        manual_output,
        latentis_output.vectors if isinstance(latentis_output, Space) else latentis_output["translated_x"],
    )

    if isinstance(A, Space):
        assert torch.allclose(
            manual_output,
            A.translate(translator=translator).vectors,
        )


def test_procrustes(
    parallel_spaces: Tuple[LatentisSpace, LatentisSpace],
):
    x, y = parallel_spaces
    x = x.vectors if isinstance(x, Space) else x
    y = y.vectors if isinstance(y, Space) else y

    procrustes = NNPipeline(
        name="procrustes",
        flows={
            "fit": Flow(name="fit", inputs=["fit_source", "fit_target"], outputs="estimator")
            .add(block="x_transform", method="fit_transform", inputs="fit_source:x", outputs="x")
            .add(block="y_transform", method="fit_transform", inputs="fit_target:x", outputs="y")
            .add(block="padder", method="fit_transform", inputs=["x", "y"], outputs=["padded_x", "padded_y"])
            .add(block="aligner", method="fit", inputs=["padded_x:x", "padded_y:y"], outputs=["aligner_state"]),
            #
            "transform": Flow(name="transform", inputs=["source"], outputs=["translated_x"])
            .add(block="x_transform", method="transform", inputs="source:x", outputs="x")
            .add(block="padder", method="transform", inputs="x", outputs="x")
            .add(block="aligner", method="transform", inputs="x", outputs="x")
            .add(block="padder", method="inverse_transform", inputs="x:y", outputs="x")
            .add(block="y_transform", method="inverse_transform", inputs="x", outputs="translated_x"),
        },
        blocks={
            "x_transform": StandardScaling(),
            "y_transform": StandardScaling(),
            "aligner": MatrixAligner("svd", align_fn_state=svd_align_state),
            "padder": ZeroPadding(),
        },
    )
    procrustes.run(flow="fit", fit_source=x, fit_target=y)
    x1 = procrustes.run(flow="transform", source=x)["translated_x"]

    manual_result = (
        ManualLatentTranslation(seed=42, centering=True, std_correction=True, l2_norm=False, method="svd")
        .fit(x=x, y=y)
        .transform(x)
    )
    _, y2 = manual_result["source"], manual_result["target"]

    assert torch.allclose(x1, y2)
