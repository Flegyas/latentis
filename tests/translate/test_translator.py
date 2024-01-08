from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import pytest
import torch

from latentis import LatentSpace
from latentis.transform.dim_matcher import ZeroPadding
from latentis.transform.translate import SVDAligner, Translator

if TYPE_CHECKING:
    from latentis.types import Space


def test_double_fitting(parallel_spaces: Tuple[Space, Space]):
    A, B = parallel_spaces

    A = A.vectors if isinstance(A, LatentSpace) else A
    B = B.vectors if isinstance(B, LatentSpace) else B

    translator = Translator(
        aligner=SVDAligner(dim_matcher=ZeroPadding()),
    )
    translator.fit(x=A, y=B)
    out1 = translator.transform(A)

    with pytest.raises(AssertionError):
        translator.fit(x=A, y=B)

    out2 = translator(A)

    if isinstance(out1, LatentSpace) and isinstance(out2, LatentSpace):
        assert out1 == out2
    else:
        assert torch.allclose(out1, out2)
