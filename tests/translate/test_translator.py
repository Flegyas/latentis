from typing import Tuple

import pytest
import torch

from latentis.estimate.dim_matcher import ZeroPadding
from latentis.estimate.orthogonal import SVDEstimator
from latentis.space import LatentSpace
from latentis.translate.translator import LatentTranslator
from latentis.types import Space


def test_double_fitting(parallel_spaces: Tuple[Space, Space]):
    A, B = parallel_spaces

    translator = LatentTranslator(
        random_seed=0,
        estimator=SVDEstimator(dim_matcher=ZeroPadding()),
    )
    translator.fit(source_data=A, target_data=B)
    out1 = translator(A)

    with pytest.raises(AssertionError):
        translator.fit(source_data=A, target_data=B)

    out2 = translator(A)

    if isinstance(out1, LatentSpace) and isinstance(out2, LatentSpace):
        assert out1 == out2
    else:
        assert torch.allclose(out1, out2)
