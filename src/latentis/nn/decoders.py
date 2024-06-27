import logging
from typing import Any, Mapping, Optional

import torch
import torch.nn.functional as F
from lightning import Trainer
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, FBetaScore, MetricCollection

from latentis.nn._base import LatentisModule

pylogger = logging.getLogger(__name__)


class SVCModel(LatentisModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return F.one_hot(torch.as_tensor(self.model.predict(x.cpu().numpy()))).to(x.device)


class LambdaModule(LatentisModule):
    def __init__(self, lambda_func) -> None:
        super().__init__()
        self.lambda_func = lambda_func

    def forward(self, x: torch.Tensor):
        return self.lambda_func(x)


class Classifier(LatentisModule):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        deep: bool,
        bias: bool = True,
        x_feature: str = "x",
        y_feature: str = "y",
        first_activation: nn.Module = nn.Tanh(),
        second_activation: nn.Module = nn.ReLU(),
        first_projection_dim: Optional[int] = None,
        trainer_params: Mapping[str, Any] = None,
        lr: float = 1e-3,
    ):
        super().__init__(
            metadata={
                "name": "classifier",
                "input_dim": input_dim,
                "num_classes": num_classes,
                "deep": deep,
                "bias": bias,
                "x_feature": x_feature,
                "y_feature": y_feature,
                "first_activation": first_activation.__class__.__name__,
                "second_activation": second_activation.__class__.__name__,
                "first_projection_dim": first_projection_dim,
                # "trainer_params": trainer_params,
                "lr": lr,
            }
        )
        self.latentis_trainer_params = dict(
            accelerator="auto",
            devices=1,
            max_epochs=5,
            logger=False,
            # callbacks=[RichProgressBar()],
            enable_progress_bar=False,
            enable_checkpointing=False,
        )
        self.latentis_trainer_params.update(trainer_params or {})
        self.lr = lr

        if not isinstance(deep, bool):
            raise ValueError(f"deep must be bool, got {deep} of type {type(deep)}")

        if not deep and (first_activation is None or second_activation is None or first_projection_dim is None):
            pylogger.warning(
                "If deep is False, first_activation, second_activation and first_projection_dim are not used!"
            )

        if callable(first_activation) and getattr(first_activation, "__name__", None) == "<lambda>":
            first_activation = LambdaModule(first_activation)

        if callable(second_activation) and getattr(second_activation, "__name__", None) == "<lambda>":
            second_activation = LambdaModule(second_activation)

        first_projection_dim = input_dim if first_projection_dim is None else first_projection_dim

        self.class_proj = (
            nn.Sequential(
                #
                nn.Linear(input_dim, first_projection_dim, bias=bias),
                first_activation,
                #
                nn.Linear(first_projection_dim, first_projection_dim // 2, bias=bias),
                second_activation,
                #
                nn.Linear(first_projection_dim // 2, num_classes, bias=bias),
            )
            if deep
            else nn.Sequential(nn.Linear(input_dim, num_classes))
        )

        self.train_metrics = MetricCollection(
            {
                "accuracy": Accuracy(task="multiclass", num_classes=num_classes),
                "f1": FBetaScore(task="multiclass", num_classes=num_classes),
            }
        )
        # self.val_metrics = self.train_metrics.clone()
        self.test_metrics = self.train_metrics.clone()

        self.x_feature: str = x_feature
        self.y_feature: str = y_feature
        self.latentis_trainer: Optional[Trainer] = None

    def forward(self, x):
        x = self.class_proj(x)
        return F.log_softmax(x, dim=1)

    def _step(self, batch, split: str):
        x = batch[self.x_feature][0]
        y = batch[self.y_feature][0]

        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        metrics = getattr(self, f"{split}_metrics")
        metrics.update(preds, y)

        self.log(f"{split}_loss", loss, prog_bar=True)
        self.log_dict(metrics, prog_bar=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch=batch, split="train")

    def test_step(self, batch, batch_idx):
        return self._step(batch=batch, split="test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def fit(self, train_dataloader: DataLoader) -> LatentisModule:
        self.latentis_trainer = Trainer(**self.latentis_trainer_params)
        self.latentis_trainer.fit(self, train_dataloaders=train_dataloader)
        return self.eval()

    def score(self, dataloader: DataLoader):
        if self.latentis_trainer is None:
            raise RuntimeError("Model has not been fitted yet!")
        return self.latentis_trainer.test(model=self, dataloaders=dataloader)
