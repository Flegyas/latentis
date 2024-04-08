from dataclasses import dataclass
from typing import Any

import torch

from latentis.benchmark.task import Task
from latentis.correspondence import Correspondence
from latentis.correspondence.correspondence import IdentityCorrespondence
from latentis.data import DATA_DIR
from latentis.data.dataset import HFDatasetView
from latentis.data.encode import EncodeTask
from latentis.data.text_encoding import HFPooler, cls_pool
from latentis.data.utils import default_collate
from latentis.nn.encoders import TextHFEncoder
from latentis.space import Space
from latentis.transform import Estimator
from latentis.transform.translate.aligner import MatrixAligner
from latentis.transform.translate.functional import svd_align_state


class Run:
    estimator: str
    #
    fit_data: str
    test_data: str
    #
    fit_correspondence: str
    test_correspondence: str


@dataclass
class EstimateResult:
    estimator: Estimator


class EstimateTask(Task):
    def __init__(self, x: Space, y: Space, pi: Correspondence, estimator: Estimator) -> None:
        super().__init__()
        self.x: Space = x
        self.y: Space = y
        self.pi: Correspondence = pi
        self.estimator: Estimator = estimator

    def _run(self) -> Estimator:
        self.estimator.set_spaces(x_space=self.x, y_space=self.y)

        x_keys, y_keys = self.pi.align(x_keys=self.x.keys, y_keys=self.y.keys)

        self.estimator.fit(x=self.x.get_vectors_by_key(keys=x_keys), y=self.y.get_vectors_by_key(keys=y_keys))

        return EstimateResult(estimator=self.estimator)


class TransformResult:
    t_space: Space


class TransformTask(Task):
    def __init__(
        self,
        space: Space,
        estimator: Estimator,
    ) -> None:
        super().__init__()
        self.space = space
        self.estimator = estimator


@dataclass
class EvaluateResult:
    metric_name: str
    score: float


class EvaluateTask(Task):
    def __init__(self, pred, target, metric) -> None:
        self.pred = pred
        self.target = target
        self.metric = metric

    def _run(self) -> Any:
        return self.metric(self.pred, self.target)


if __name__ == "__main__":
    # EncodeTask
    dataset = HFDatasetView.load_from_disk(DATA_DIR / "trec")
    device = torch.device("cpu")

    x_fit_task = EncodeTask(
        dataset=dataset,
        split="train",
        feature="text",
        model=TextHFEncoder("bert-base-cased"),
        collate_fn=default_collate,
        encoding_batch_size=256,
        num_workers=1,
        save_source_model=False,
        pooler=HFPooler(layers=[12], pooling_fn=cls_pool, output_dim=768),
        device=device,
    )
    y_fit_task = EncodeTask(
        dataset=dataset,
        split="train",
        feature="text",
        model=TextHFEncoder("roberta-base"),
        collate_fn=default_collate,
        encoding_batch_size=256,
        num_workers=1,
        save_source_model=False,
        pooler=HFPooler(layers=[12], pooling_fn=cls_pool, output_dim=768),
        device=device,
    )

    x_test_task = EncodeTask(
        dataset=dataset,
        split="test",
        feature="text",
        model=TextHFEncoder("bert-base-cased"),
        collate_fn=default_collate,
        encoding_batch_size=256,
        num_workers=1,
        save_source_model=False,
        pooler=HFPooler(layers=[12], pooling_fn=cls_pool, output_dim=768),
        device=device,
    )
    y_test_task = EncodeTask(
        dataset=dataset,
        split="test",
        feature="text",
        model=TextHFEncoder("roberta-base"),
        collate_fn=default_collate,
        encoding_batch_size=256,
        num_workers=1,
        save_source_model=False,
        pooler=HFPooler(layers=[12], pooling_fn=cls_pool, output_dim=768),
        device=device,
    )

    x_fit, y_fit = x_fit_task.run().space, y_fit_task.run().space
    x_test, y_test = x_test_task.run().space, y_test_task.run().space

    # assert len(x_fit) == len(y_fit)
    # assert len(x_test) == len(y_test)

    fit_pi = IdentityCorrespondence(n_samples=len(x_fit))
    test_pi = IdentityCorrespondence(n_samples=len(x_test))

    estimate_task = EstimateTask(
        x=x_fit, y=y_fit, pi=fit_pi, estimator=MatrixAligner(name="orthogonal", align_fn_state=svd_align_state)
    )
    print(estimate_task.run().estimator)
    # transform_task = TransformTask("space", "estimator")
    # evaluate_task = EvaluateTask("pred", "target", "metric")
