import itertools
import json
from collections import defaultdict
from dataclasses import dataclass, field
from enum import auto
from pathlib import Path
from typing import Callable, Iterable, Mapping, Optional, Sequence, Tuple

import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from latentis.benchmark import BENCHMARK_DIR
from latentis.benchmark.correspondence import Correspondence, SameDatasetCorrespondence
from latentis.data import DATA_DIR
from latentis.data.dataset import LatentisDataset
from latentis.measure import PairwiseMetric
from latentis.space import LatentSpace
from latentis.space._base import EncodingKey
from latentis.transform.translate.aligner import Aligner, MatrixAligner
from latentis.transform.translate.functional import svd_align_state
from latentis.types import StrEnum


class TaskProperty(StrEnum):
    pass


class TaskType(StrEnum):
    CLASSIFICATION = auto()
    AUTOENCODING = auto()


@dataclass(frozen=True)
class Task:
    col_name: str
    task_type: TaskType
    properties: Mapping[TaskProperty, str] = field(default_factory=lambda: {})


@dataclass(frozen=True)
class EncodingKeychain:
    fit: EncodingKey
    test: EncodingKey

    @staticmethod
    def build(
        fit_split: str,
        test_split: str,
        dataset: str,
        feature: str,
        model_name: str,
        pooler_name: str,
        **extra_properties,
    ) -> "EncodingKeychain":
        split2key = {
            split_name: EncodingKey(
                dataset=dataset,
                feature=feature,
                split=split,
                model_name=model_name,
                pooler_name=pooler_name,
                **extra_properties,
            )
            for split_name, split in zip(("fit", "test"), [fit_split, test_split])
        }

        return EncodingKeychain(**split2key)


class Benchmark:
    def __init__(self, name: str, tasks: Sequence[Task]):
        self.name = name
        self.tasks = tasks


class L2Similarity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return (x - y).norm(p=2, dim=-1).mean()

    @property
    def name(self):
        return "l2"


class L1Similarity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return (x - y).norm(p=1, dim=-1).mean()

    @property
    def name(self):
        return "l1"


class CosineSimilarity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return F.cosine_similarity(x, y, dim=-1).mean()

    @property
    def name(self):
        return "cosine"


def exhaustive_policy(x: EncodingKeychain, y: EncodingKeychain) -> bool:
    return True


class AlignmentTask:
    def __init__(
        self,
        name: str,
        keychains: Sequence[EncodingKey],
        correspondence: Correspondence,
        fit_metrics: Sequence[PairwiseMetric],
        test_metrics: Sequence[PairwiseMetric],
        pairing_policy: Callable[
            [Sequence[LatentSpace], Sequence[LatentSpace]], Iterable[Tuple[LatentSpace, LatentSpace]]
        ] = exhaustive_policy,
    ):
        self.name: str = name
        self.keychains: Sequence[EncodingKeychain] = keychains
        self.correspondence: Correspondence = correspondence
        self.fit_metrics = fit_metrics
        self.test_metrics = test_metrics
        self.pairing_policy = pairing_policy

    def get_pairs(self) -> Iterable[Tuple[LatentSpace, LatentSpace]]:
        yield from (pair for pair in itertools.product(self.keychains, repeat=2) if self.pairing_policy(*pair))

    def get_results(self) -> None:
        benchmark_dir = DATA_DIR / self.name

        result = defaultdict(dict)

        for aligner_id in benchmark_dir.iterdir():
            for run_id in aligner_id.iterdir():
                result[aligner_id][run_id] = pd.read_csv(run_id, sep="\t")

        return result

    def evaluate(self, estimator: Aligner, run_id: Optional[str] = None, root_dir: Path = BENCHMARK_DIR) -> None:
        # TODO
        # if root_dir is not None:
        #     run_id = run_id or time.strftime("%Y%m%d-%H%M%S")
        #     run_path: Path = root_dir / self.name / aligner.name / f"{run_id}.tsv"
        #     try:
        #         run_path.parent.mkdir(parents=True, exist_ok=False)
        #     except FileExistsError:
        #         raise FileExistsError(
        #             f"Run {run_id} already exists for benchmark {self.name} and aligner {aligner.name}"
        #         )

        results = defaultdict(list)

        for x, y in self.get_pairs():
            x: EncodingKeychain
            y: EncodingKeychain

            x_fit: torch.Tensor = self.correspondence.get_fit_vectors(x.fit)
            y_fit: torch.Tensor = self.correspondence.get_fit_vectors(y.fit)

            estimator = estimator.fit(x=x_fit, y=y_fit)

            fit_estimation = estimator(x=x_fit, y=y_fit)

            # TODO: apply optional decoder here to enable stitching evaluation

            fit_pair_stats = {
                f"fit_{metric.name}": metric(fit_estimation["x"], fit_estimation["y"]) for metric in self.fit_metrics
            }

            x_test: torch.Tensor = self.correspondence.get_test_vectors(x.test)
            y_test: torch.Tensor = self.correspondence.get_test_vectors(y.test)

            test_estimation = estimator(x=x_test, y=y_test)

            # TODO: apply optional decoder here to enable stitching evaluation

            test_pair_stats = {
                f"test_{metric.name}": metric(test_estimation["x"], test_estimation["y"])
                for metric in self.test_metrics
            }

            pair_stats = {**fit_pair_stats, **test_pair_stats}

            pair_stats["x_fit_key"] = x.fit
            pair_stats["y_fit_key"] = y.fit

            for k, v in pair_stats.items():
                results[k].append(v.item() if isinstance(v, torch.Tensor) else v)

        # if root_dir is not None:
        #     results.to_csv(run_path, sep="\t", index=False)

        return results


if __name__ == "__main__":
    dataset: LatentisDataset = LatentisDataset.load_from_disk(DATA_DIR / "imdb")
    print(dataset)

    task = AlignmentTask(
        name="align_imdb",
        keychains=(
            [
                EncodingKeychain.build(
                    fit_split="train",
                    test_split="test",
                    dataset="imdb",
                    feature="text",
                    model_name="bert-base-cased",
                    pooler_name="cls_pool_12",
                ),
                EncodingKeychain.build(
                    fit_split="train",
                    test_split="test",
                    dataset="imdb",
                    feature="text",
                    model_name="bert-base-uncased",
                    pooler_name="cls_pool_12",
                ),
            ]
        ),
        pairing_policy=exhaustive_policy,
        correspondence=SameDatasetCorrespondence(dataset=dataset, perc=1, seed=42),
        fit_metrics=[L2Similarity(), L1Similarity(), CosineSimilarity()],
        test_metrics=[L2Similarity(), L1Similarity(), CosineSimilarity()],
    )
    print(task)

    result = task.evaluate(MatrixAligner(name="svd", align_fn_state=svd_align_state))
    print(json.dumps(result, indent=4))
