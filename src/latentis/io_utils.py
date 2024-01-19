from pathlib import Path

import torch
from torch import nn


# TODO: Handle versioning
def save_model(model: nn.Module, target_path: Path, version: int):
    torch.save(model, target_path)


def load_model(model_path: Path, version: int) -> nn.Module:
    return torch.load(model_path)
