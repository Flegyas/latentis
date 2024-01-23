import logging
from typing import Any, Mapping, Optional

import torch
import torch.nn.functional as F
from lightning import Trainer
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, FBetaScore, MetricCollection

from latentis.data import DATA_DIR
from latentis.data.dataset import LatentisDataset
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

        super().__init__(model_key="classifier")
        self.trainer_params = dict(
            accelerator="auto",
            devices=1,
            max_epochs=5,
            logger=False,
            # callbacks=[RichProgressBar()],
            enable_progress_bar=False,
            enable_checkpointing=False,
        )
        self.trainer_params.update(trainer_params or {})
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
        self.trainer: Optional[Trainer] = None

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
        self.trainer = Trainer(**self.trainer_params)
        self.trainer.fit(self, train_dataloaders=train_dataloader)
        return self.eval()

    def score(self, dataloader: DataLoader):
        if self.trainer is None:
            raise RuntimeError("Model has not been fitted yet!")
        return self.trainer.test(dataloaders=dataloader)


if __name__ == "__main__":
    dataset = LatentisDataset.load_from_disk(DATA_DIR / "imdb")

    nclasses = len(dataset.hf_dataset["train"].features["label"].names)

    raw_data = dataset.hf_dataset
    label_key = "label"

    dataset.encodings.get_item(item_key="3")

    model = Classifier(
        input_dim=dataset.encodings.load_item(item_key="3").shape[1],
        num_classes=nclasses,
        deep=True,
        bias=True,
        x_feature="encodings_key",
        y_feature="hf_y_keys",
        first_activation=nn.Tanh(),
        second_activation=nn.ReLU(),
        first_projection_dim=None,
    )
    device = "cpu" if not torch.cuda.is_available() else "cuda"
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss = nn.CrossEntropyLoss()

    train_dataloader = DataLoader(
        dataset.get_dataset_view(
            split="train",
            encodings_key=dataset.encodings.get_item_key(split="train", model="bert-base-uncased"),
            hf_x_keys=None,
            hf_y_keys=("label",),
        ),
        batch_size=32,
        shuffle=True,
    )

    test_dataloader = DataLoader(
        dataset.get_dataset_view(
            split="test",
            encodings_key=dataset.encodings.get_item_key(split="test", model="bert-base-uncased"),
            hf_x_keys=None,
            hf_y_keys=("label",),
        ),
        batch_size=32,
        shuffle=True,
    )

    model.fit(train_dataloader=train_dataloader)
    print(model.score(dataloader=test_dataloader))
