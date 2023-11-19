from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import pytest
import torch
import torch.nn.functional as F

from latentis.measure import MetricFn
from latentis.measure.base import CKA, CKAMode

if TYPE_CHECKING:
    from latentis.types import Space

TOL = 1e-6


@pytest.mark.parametrize(
    "metric_fn",
    [
        (F.cosine_similarity),
        (F.mse_loss),
        (F.l1_loss),
        (lambda x, y: torch.trace(torch.matmul(x, y.t()))),
    ],
)
def test_metric(metric_fn: Callable[[Space, Space], torch.Tensor], same_shape_spaces):
    space1, space2 = same_shape_spaces
    fn_result = metric_fn(
        space1 if isinstance(space1, torch.Tensor) else space1.vectors,
        space2 if isinstance(space2, torch.Tensor) else space2.vectors,
    )

    obj_result = MetricFn(key="test", fn=metric_fn)(space1, space2)["test"]

    assert torch.allclose(fn_result, obj_result)


@pytest.mark.parametrize(
    "mode",
    [
        (CKAMode.LINEAR),
        (CKAMode.RBF),
    ],
)
def test_cka(mode: CKAMode, same_shape_spaces, different_dim_spaces):
    for spaces in [same_shape_spaces, different_dim_spaces]:
        space1, space2 = spaces
        cka = CKA(mode=mode)
        cka_result = cka(space1, space2)

        # cka must stay in 0, 1 range
        assert 0.0 - TOL <= cka_result <= 1.0 + TOL

        # cka is symmetric
        symm_cka_result = cka(space2, space1)
        assert symm_cka_result == pytest.approx(cka_result, abs=TOL)

        # check that GPU works correctly
        # cka_gpu = CKA(mode=mode, device=torch.device("cuda"))
        # cka_result = cka_gpu(space1, space2)

        # assert cka_result.device.type == "cuda"
