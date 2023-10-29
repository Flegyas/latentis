from typing import Tuple

import pytest
import torch

from latentis.estimate.orthogonal import SVDEstimator
from latentis.space import LatentSpace
from latentis.translate.translator import LatentTranslator


def test_double_fitting(parallel_spaces: Tuple[LatentSpace, LatentSpace]):
    space1, space2 = parallel_spaces
    A = space1.vectors
    B = space2.vectors

    translator = LatentTranslator(
        random_seed=0,
        estimator=SVDEstimator(),
    )
    translator.fit(source_data=A, target_data=B)
    out1 = translator(A)["target"]

    with pytest.raises(AssertionError):
        translator.fit(source_data=A, target_data=B)

    out2 = translator(A)["target"]

    assert torch.allclose(out1, out2)
