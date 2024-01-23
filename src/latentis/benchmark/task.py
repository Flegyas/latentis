import itertools
from dataclasses import dataclass
from enum import auto
from typing import Mapping, Optional, Sequence

import pandas as pd
import torch
import torch.nn.functional as F
from datasets.search import IndexableMixin
from torch import nn

from latentis.benchmark.correspondence import Correspondence, IdentityCorrespondence
from latentis.data import DATA_DIR
from latentis.data.dataset import LatentisDataset
from latentis.measure import PairwiseMetric
from latentis.nn import LatentisModule
from latentis.space import LatentSpace
from latentis.transform import Estimator
from latentis.transform.translate.aligner import MatrixAligner
from latentis.transform.translate.functional import svd_align_state
from latentis.types import Properties, StrEnum


class TaskProperty(StrEnum):
    pass


class TaskType(StrEnum):
    CLASSIFICATION = auto()
    AUTOENCODING = auto()


class Accuracy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return (x.argmax(dim=-1) == y).float().mean()

    @property
    def item_id(self):
        return "Accuracy"

    @property
    def properties(self):
        return {"name": self.item_id}


class L2Similarity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return (x - y).norm(p=2, dim=-1).mean()

    @property
    def item_id(self):
        return "L1Similarity"

    @property
    def properties(self):
        return {"name": self.item_id}


class L1Similarity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return (x - y).norm(p=1, dim=-1).mean()

    @property
    def item_id(self):
        return "L1Similarity"

    @property
    def properties(self):
        return {"name": self.item_id}


class CosineSimilarity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return F.cosine_similarity(x, y, dim=-1).mean()

    @property
    def item_id(self):
        return "cosine"

    @property
    def properties(self):
        return {"name": self.item_id}


@dataclass(frozen=True)
class ExperimentResult:
    x_fit: str
    y_fit: str
    fit_correspondence: str
    #
    estimator: str
    #
    x_test: str
    y_test: str
    test_correspondence: str
    #
    metric: str
    score: float
    #
    # y_fit_decoder: str
    # y_gt: str
    # downstream_metric: str
    # downstream_score: float


class Experiment(IndexableMixin):
    @property
    def properties(self) -> Properties:
        return {
            "x_fit": self.x_fit.properties,
            "y_fit": self.y_fit.properties,
            "fit_correspondence": self.fit_correspondence.properties,
            #
            "estimator": self.estimator.properties,
            #
            "x_test": self.x_test.properties,
            "y_test": self.y_test.properties,
            "test_correspondence": self.test_correspondence.properties,
            #
            "metric": {
                metric_id: metric.properties
                for metric_id, metric in itertools.chain(self.latent_metrics.items(), self.downstream_metrics.items())
            },
            #
            # "y_fit_decoder": self.y_fit_decoder.properties if self.y_fit_decoder is not None else {},
            # "y_gt": self.y_gt.properties if self.y_gt is not None else {},
        }

    def __init__(
        self,
        x_fit: LatentSpace,
        y_fit: LatentSpace,
        fit_correspondence: Correspondence,
        #
        estimator: Estimator,
        #
        x_test: LatentSpace,
        y_test: LatentSpace,
        test_correspondence: Correspondence,
        #
        latent_metrics: Sequence[PairwiseMetric],
        #
        y_fit_decoder: Optional[nn.Module] = None,
        y_gt: Optional[torch.Tensor] = None,
        downstream_metrics: Optional[Sequence[PairwiseMetric]] = None,
    ):
        if not ((y_fit_decoder is None) == (y_gt is None) == (downstream_metrics is None)):
            raise ValueError("Either all or none of the downstream task arguments must be provided")

        self.x_fit = x_fit
        self.y_fit = y_fit
        self.fit_correspondence = fit_correspondence

        self.estimator = estimator

        self.x_test = x_test
        self.y_test = y_test
        self.test_correspondence = test_correspondence

        latent_metrics = [] if latent_metrics is None else latent_metrics
        self.latent_metrics = {latent_metrics.item_id: latent_metrics for latent_metrics in latent_metrics}

        self.y_fit_decoder = y_fit_decoder
        self.y_gt = y_gt

        downstream_metrics = [] if downstream_metrics is None else downstream_metrics
        self.downstream_metrics = {
            downstream_metric.item_id: downstream_metric for downstream_metric in downstream_metrics
        }

    def estimation_stage(self):
        corresponding_x_fit: torch.Tensor = self.fit_correspondence.get_x_ids()
        corresponding_y_fit: torch.Tensor = self.fit_correspondence.get_y_ids()

        estimator = self.estimator.set_spaces(x_space=self.x_fit, y_space=self.y_fit)
        estimator = estimator.fit(x=self.x_fit[corresponding_x_fit], y=self.y_fit[corresponding_y_fit])

        return estimator

    def transform_stage(self, estimator: Estimator):
        corresponding_x_test: torch.Tensor = self.test_correspondence.get_x_ids()
        corresponding_y_test: torch.Tensor = self.test_correspondence.get_y_ids()

        x_test_transformed, y_test_transformed = estimator.transform(
            x=self.x_test[corresponding_x_test], y=self.y_test[corresponding_y_test]
        )
        return x_test_transformed, y_test_transformed

    def latent_evaluation_stage(self, x_test_transformed: torch.Tensor, y_test_transformed: torch.Tensor):
        return {
            metric_id: metric(x_test_transformed, y_test_transformed).item()
            for metric_id, metric in self.latent_metrics.items()
        }

    def downstream_evaluation_stage(self, x_test_transformed: torch.Tensor):
        if self.y_fit_decoder is not None:
            y_fit_decoder: LatentisModule = self.y_fit.decoders.load_item(**self.y_fit_decoder)

            return {
                metric_id: metric(y_fit_decoder(x_test_transformed), self.y_gt).item()
                for metric_id, metric in self.downstream_metrics.items()
            }
        else:
            return {}

    def export(self, results: Sequence[ExperimentResult], selection: Mapping[str, Sequence[str]]) -> pd.DataFrame:
        experiment_properties = self.properties
        export = {f"{k}/{v}": [] for k, values in selection.items() for v in values}
        export.update(**{k: [] for k, values in selection.items() if len(values) == 0})

        for result in results:
            for selection_key, selection_values in selection.items():
                for selection_value in selection_values:
                    k = f"{selection_key}/{selection_value}"
                    if selection_key != "metric":
                        export[k].append(experiment_properties[selection_key][selection_value])
                    else:
                        export[k].append(
                            experiment_properties[selection_key][getattr(result, selection_key)][selection_value]
                        )
                if len(selection_values) == 0:
                    export[selection_key].append(getattr(result, selection_key))

        return pd.DataFrame.from_dict(export)

    def run(
        self,
        estimator: Optional[Estimator] = None,
    ):
        estimator = self.estimation_stage() if estimator is None else estimator

        x_test_transformed, y_test_transformed = self.transform_stage(estimator=estimator)

        latent_evaluation = self.latent_evaluation_stage(
            x_test_transformed=x_test_transformed, y_test_transformed=y_test_transformed
        )
        downstream_evaluation = self.downstream_evaluation_stage(x_test_transformed=x_test_transformed)

        for metric, score in itertools.chain(latent_evaluation.items(), downstream_evaluation.items()):
            yield ExperimentResult(
                x_fit=self.x_fit.item_id,
                y_fit=self.y_fit.item_id,
                fit_correspondence=self.fit_correspondence.item_id,
                #
                estimator=estimator.item_id,
                #
                x_test=self.x_test.name,
                y_test=self.y_test.name,
                test_correspondence=self.test_correspondence.item_id,
                #
                metric=metric,
                score=score,
                #
                # y_fit_decoder=self.y_fit_decoder.name,
                # y_gt=self.y_gt.name,
                # downstream_metric=downstream_metric,
                # downstream_score=downstream_score,
            )


if __name__ == "__main__":
    dataset: LatentisDataset = LatentisDataset.load_from_disk(DATA_DIR / "trec")

    print(dataset)
    # correspondence_index = CorrespondenceIndex.load_from_disk(DATA_DIR / "correspondence")

    x_fit = dataset.encodings.load_item(
        **{
            "split": "train",
            "model/hf_name": "bert-base-cased",
            "pool": "cls",
            "layer": 12,
        }
    )
    y_fit = dataset.encodings.load_item(
        **{
            "split": "train",
            "model/hf_name": "bert-base-cased",
            "pool": "cls",
            "layer": 12,
        }
    )
    x_test = dataset.encodings.load_item(
        **{
            "split": "test",
            "model/hf_name": "bert-base-cased",
            "pool": "cls",
            "layer": 12,
        }
    )
    y_test = dataset.encodings.load_item(
        **{
            "split": "test",
            "model/hf_name": "bert-base-uncased",
            "pool": "cls",
            "layer": 12,
        }
    )

    experiment = Experiment(
        x_fit=x_fit,
        y_fit=y_fit,
        fit_correspondence=IdentityCorrespondence(
            n_samples=len(dataset.hf_dataset["train"]),
        ),  # .random_subset(factor=0.01, seed=42),
        x_test=x_test,
        y_test=y_test,  # .random_subset(factor=0.01, seed=42),
        test_correspondence=IdentityCorrespondence(
            n_samples=len(dataset.hf_dataset["test"]),
        ),
        estimator=MatrixAligner(name="svd", align_fn_state=svd_align_state),
        latent_metrics=[L2Similarity(), L1Similarity(), CosineSimilarity()],
        y_fit_decoder={},
        y_gt=torch.as_tensor(dataset.hf_dataset["test"]["coarse_label"]),
        downstream_metrics=[Accuracy()],
    )

    print(experiment.properties)

    runs = list(experiment.run())
    print(runs[0])

    space_selection = ["dataset", "model/hf_name", "pool", "layer"]
    selection = {
        "x_fit": space_selection,
        "y_fit": space_selection,
        "estimator": ["name"],
        "x_test": space_selection,
        "y_test": space_selection,
        "metric": ["name"],
        "score": [],
    }

    experiment.export(results=runs, selection=selection).to_csv(DATA_DIR / "experiment.csv", index=False, sep="\t")
