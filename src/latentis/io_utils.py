import json
from pathlib import Path
from typing import Callable, Optional

import torch
from torch import nn


# TODO: Handle versioning
def save_model(model: nn.Module, target_path: Path, version: int):
    torch.save(model, target_path)


def load_model(model_path: Path, version: int) -> nn.Module:
    return torch.load(model_path)


def _default_json(o):
    return o.__dict__


def save_json(
    obj: object,
    path: Path,
    indent: Optional[int] = 4,
    sort_keys: bool = True,
    default: Optional[Callable] = _default_json,
):
    with open(path, "w", encoding="utf-8") as fw:
        json.dump(obj, fw, indent=indent, sort_keys=sort_keys, default=default)


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as fr:
        return json.load(fr)
