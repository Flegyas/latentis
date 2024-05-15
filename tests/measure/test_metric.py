from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import pytest
import torch
import torch.nn.functional as F

from latentis.measure import MetricFn
from latentis.measure.cka import CKA, CKAMode
from latentis.measure.functional.cka import cka as cka_fn
from latentis.measure.functional.cka import kernel_hsic, linear_hsic
from latentis.measure.functional.svcca import robust_svcca as svcca_fn
from latentis.measure.svcca import SVCCA
from latentis.space import LatentSpace

if TYPE_CHECKING:
    from latentis.types import Space


TOL = 1e-3
SVCCA_TOLERANCE = 1e-3


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
def test_cka(mode: CKAMode, same_shape_spaces, different_dim_spaces, precomputed_cka):
    # test object-oriented interface
    space1, space2 = same_shape_spaces[0], same_shape_spaces[1]

    cka_none = CKA(mode=mode, device=None)
    cka_result = cka_none(space1, space2)

    assert cka_result.device.type == "cpu"

    # check that GPU works correctly
    if torch.cuda.is_available():
        cka_gpu = CKA(mode=mode, device=torch.device("cuda"))
        cka_result = cka_gpu(space1, space2)

        assert cka_result.device.type == "cuda"

        cka_gpu = cka_none.to("cuda")
        cka_result = cka_gpu(space1, space2)

        assert cka_result.device.type == "cuda"

    for spaces in [same_shape_spaces, different_dim_spaces]:
        space1, space2 = spaces
        cka = CKA(mode=mode)

        # check that CKA is 1 for identical spaces
        cka_result = cka(space1, space1)
        assert cka_result == pytest.approx(1.0, abs=TOL)

        # cka must stay in 0, 1 range
        cka_result = cka(space1, space2)
        assert 0.0 - TOL <= cka_result <= 1.0 + TOL

        # cka is symmetric
        symm_cka_result = cka(space2, space1)
        assert symm_cka_result == pytest.approx(cka_result, abs=TOL)

    # check that the cka results didn't change from stored computations
    cka_result = CKA(mode=mode, device="cpu")(precomputed_cka["stored_space1"], precomputed_cka["stored_space2"])

    # higher tolerance because of the RBF kernel being noisy
    assert cka_result == pytest.approx(precomputed_cka[mode], abs=1e-4)

    # test functional interface
    space1, space2 = same_shape_spaces[0], same_shape_spaces[1]
    hsic = linear_hsic if mode == CKAMode.LINEAR else kernel_hsic
    cka_result = cka_fn(space1, space2, hsic=hsic)

    assert cka_result == pytest.approx(CKA(mode=mode)(space1, space2), abs=TOL)
    assert cka_result.device.type == "cpu"

    if torch.cuda.is_available():
        if isinstance(space1, torch.Tensor):
            space1 = space1.to("cuda")
            space2 = space2.to("cuda")
        else:
            space1 = LatentSpace.like(space1, vector_source=space1.vectors.to("cuda"))
            space2 = LatentSpace.like(space2, vector_source=space2.vectors.to("cuda"))

        cka_result = cka_fn(space1, space2, hsic=hsic)
        assert cka_result.device.type == "cuda"


def test_svcca(same_shape_spaces, different_dim_spaces, precomputed_svcca):
    # test object-oriented interface
    space1, space2 = same_shape_spaces[0], same_shape_spaces[1]

    svcca_none = SVCCA(device=None, tolerance=SVCCA_TOLERANCE)
    svcca_result = svcca_none(space1, space2)

    assert svcca_result.device.type == "cpu"

    # check that GPU works correctly
    if torch.cuda.is_available():
        svcca_gpu = SVCCA(device=torch.device("cuda"), tolerance=SVCCA_TOLERANCE)
        svcca_result = svcca_gpu(space1, space2)

        assert svcca_result.device.type == "cuda"

        svcca_gpu = svcca_none.to("cuda")
        svcca_result = svcca_gpu(space1, space2)

        assert svcca_result.device.type == "cuda"

    for spaces in [same_shape_spaces, different_dim_spaces]:
        space1, space2 = spaces
        svcca = SVCCA(tolerance=SVCCA_TOLERANCE)

        svcca_result = svcca(space1, space1)
        assert svcca_result == pytest.approx(
            1.0, abs=TOL
        ), f"Computed a SVCCA value of {svcca_result} for identical spaces while it should be 1. "

        # svcca must stay in 0, 1 range
        svcca_result = svcca(space1, space2)
        assert 0.0 - TOL <= svcca_result <= 1.0 + TOL

        symm_svcca_result = svcca(space2, space1)
        assert symm_svcca_result == pytest.approx(
            svcca_result, abs=TOL
        ), f"Computed asymmetric SVCCA values: {symm_svcca_result}, {svcca_result} "

    # check that the svcca results didn't change from stored computations
    svcca_result = SVCCA(device="cpu", tolerance=SVCCA_TOLERANCE)(
        precomputed_svcca["stored_space1"], precomputed_svcca["stored_space2"]
    )

    # higher tolerance because of the RBF kernel being noisy
    assert svcca_result == pytest.approx(precomputed_svcca["result"], abs=1e-3)

    # test functional interface
    space1, space2 = same_shape_spaces[0], same_shape_spaces[1]
    svcca_result = svcca_fn(space1, space2, tolerance=SVCCA_TOLERANCE)

    assert svcca_result == pytest.approx(SVCCA(tolerance=SVCCA_TOLERANCE)(space1, space2), abs=TOL)
    assert svcca_result.device.type == "cpu"

    if torch.cuda.is_available():
        if isinstance(space1, torch.Tensor):
            space1 = space1.to("cuda")
            space2 = space2.to("cuda")
        else:
            space1 = LatentSpace.like(space1, vector_source=space1.vectors.to("cuda"))
            space2 = LatentSpace.like(space2, vector_source=space2.vectors.to("cuda"))

        svcca_result = svcca_fn(space1, space2, tolerance=SVCCA_TOLERANCE)
        assert svcca_result.device.type == "cuda"
