from typing import Tuple

import pytest
import torch

from latentis.estimate.orthogonal import SVDEstimator
from latentis.space import LatentSpace
from latentis.translate.translator import LatentTranslator


def test_double_fitting(parallel_spaces: Tuple[LatentSpace, LatentSpace]):
    A, B = parallel_spaces

    translator = LatentTranslator(
        random_seed=0,
        estimator=SVDEstimator(),
    )
    translator.fit(source_data=A, target_data=B)
    out1 = translator(A).vectors

    with pytest.raises(AssertionError):
        translator.fit(source_data=A, target_data=B)

    out2 = translator(A).vectors

    assert torch.allclose(out1, out2)
