from typing import List, Sequence

import torch
from PIL.Image import Image


@torch.no_grad()
def image_encode(
    images: Sequence[Image],
    transform,
):
    images: List[torch.Tensor] = [transform(image["image"].convert("RGB")) for image in images]
    images: torch.Tensor = torch.stack(images, dim=0)

    return {"image": images}


@torch.no_grad()
def clip_image_encode(
    images: Sequence[Image],
    transform,
):
    images = [image["image"].convert("RGB") for image in images]
    images = transform(images=images, return_tensors="pt")
    return {"image": images}
