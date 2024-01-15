import torch
import torch.nn.functional as F
from torch import nn

from latentis.types import MetadataMixin, SerializableMixin


class SVCModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return F.one_hot(torch.as_tensor(self.model.predict(x.cpu().numpy()))).to(x.device)


class Encoder(nn.Module, SerializableMixin, MetadataMixin):
    pass


class Decoder(nn.Module, SerializableMixin, MetadataMixin):
    pass
