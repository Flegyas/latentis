import pytest
import torch
from transformers import PreTrainedTokenizerBase, BatchEncoding
from latentis.nn.encoders import TextHFEncoder


@pytest.mark.parametrize("hf_name", ["bert-base-uncased", "roberta-base"])
@pytest.mark.parametrize("requires_grad", [True, False])
def test_text_hfencoder_init(hf_name, requires_grad):
    encoder = TextHFEncoder(hf_name=hf_name, requires_grad=requires_grad)
    assert isinstance(encoder.tokenizer, PreTrainedTokenizerBase)
    # Check if the model's parameters have requires_grad set as expected
    for param in encoder.model.parameters():
        assert param.requires_grad == requires_grad
    assert encoder.output_dim is not None


@pytest.mark.parametrize("hf_name", ["bert-base-uncased"])
def test_text_hfencoder_pre_encode_and_encode(hf_name):
    encoder = TextHFEncoder(hf_name=hf_name)

    # Sample input
    samples = [{"text": "Hello world!"}]
    feature = "text"

    # Call pre_encode
    pre_encoded = encoder.pre_encode(samples, feature)

    # Assert that pre_encoded is a BatchEncoding instance
    assert isinstance(pre_encoded, BatchEncoding)

    # Check if the 'tok_out' contains the necessary elements
    assert "tok_out" in pre_encoded
    assert "input_ids" in pre_encoded["tok_out"]
    assert "attention_mask" in pre_encoded["tok_out"]
    assert "special_tokens_mask" in pre_encoded["tok_out"]

    # Call the encode method
    special_tokens_mask = pre_encoded["tok_out"]["special_tokens_mask"]
    encoded = encoder.encode(pre_encoded)

    # Check the structure of the output from encode
    assert isinstance(encoded, dict)
    assert "x" in encoded
    assert "mask" in encoded

    # Check if 'x' is a tuple of tensors
    assert isinstance(encoded["x"], tuple)
    for tensor in encoded["x"]:
        assert isinstance(tensor, torch.Tensor)  # Ensure each item in 'x' is a tensor

    # Check if 'mask' is a tensor
    assert isinstance(encoded["mask"], torch.Tensor)

    # Ensure special tokens are not active in the mask (where mask should be 0 for special tokens)
    assert torch.all(encoded["mask"][special_tokens_mask == 1] == 0)

    # Optionally check tensor shapes
    input_shape = pre_encoded["tok_out"]["input_ids"].shape
    assert encoded["x"][0].shape[0] == input_shape[0]  # Batch size should match
    assert encoded["mask"].shape == pre_encoded["tok_out"]["attention_mask"].shape
