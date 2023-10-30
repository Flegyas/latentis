import pytest
import torch

from latentis.transforms import L2, Centering, Independent, StandardScaling, STDScaling


@pytest.mark.parametrize(
    "transform_type",
    [
        Centering,
        L2,
        STDScaling,
        StandardScaling,
    ],
)
def test_functional_transforms(
    transform_type: Independent,
    tensor_space_with_ref,
):
    space, reference = tensor_space_with_ref

    transform = transform_type()
    transform.fit(reference=reference)

    out1 = transform(x=space)
    out2 = transform(x=space, reference=reference)
    out3 = transform_type()(x=space, reference=reference)

    assert torch.allclose(out1, out2)
    assert torch.allclose(out1, out3)

    rev_out1 = transform.reverse(x=out1)
    rev_out2 = transform.reverse(x=out2, reference=reference)
    rev_out3 = transform_type().reverse(x=out3, reference=reference)

    assert torch.allclose(rev_out1, rev_out2)
    assert torch.allclose(rev_out1, rev_out3)
