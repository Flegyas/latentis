from typing import Callable

import pytest
import torch
import torch.nn.functional as F

from latentis.measure import MetricFn
from latentis.types import Space


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
