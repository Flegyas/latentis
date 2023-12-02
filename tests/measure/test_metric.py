from typing import Callable

import pytest
import torch
import torch.nn.functional as F

from latentis.measure import MetricFn
from latentis.types import Space
from latentis.measure.cka import CKA, CKAMode

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

    # test object-oriented interface
    space1, space2 = same_shape_spaces[0], same_shape_spaces[1]

    # check that GPU works correctly
    cka_gpu = CKA(mode=mode, device=torch.device("cuda"))
    cka_result = cka_gpu(space1, space2)

    assert cka_result.device.type == "cuda"

    cka_none = CKA(mode=mode, device=None)
    cka_result = cka_none(space1, space2)

    assert cka_result.device.type == "cpu"

    cka_gpu = cka_none.to('cuda')
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

    # test functional interface
    space1, space2 = same_shape_spaces[0], same_shape_spaces[1]
    cka_result = CKA(mode=mode, device='cpu')(space1, space2)

    assert cka_result == pytest.approx(CKA(mode=mode)(space1, space2), abs=TOL)
    assert cka_result.device.type == 'cpu'
    
    cka_result = CKA(mode=mode, device='cuda')(space1, space2)
    assert cka_result.device.type == 'cuda'

    cka_result = CKA(mode=mode, device=None)(space1, space2)
    assert cka_result.device.type == 'cpu'    














