from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import pytest

if TYPE_CHECKING:
    from latentis.types import Space


def test_double_fitting(parallel_spaces: Tuple[Space, Space]):
    pytest.skip("This test is not yet refactored.")
    # A, B = parallel_spaces

    # translator = LatentTranslator(
    #     random_seed=0,
    #     estimator=SVDEstimator(dim_matcher=ZeroPadding()),
    # )
    # translator.fit(source_data=A, target_data=B)
    # out1 = translator(A)

    # with pytest.raises(AssertionError):
    #     translator.fit(source_data=A, target_data=B)

    # out2 = translator(A)

    # if isinstance(out1, LatentSpace) and isinstance(out2, LatentSpace):
    #     assert out1 == out2
    # else:
    #     assert torch.allclose(out1, out2)
