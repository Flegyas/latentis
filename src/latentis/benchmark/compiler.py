from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional, Tuple

from omegaconf import DictConfig

from latentis.benchmark import BENCHMARK_DIR


class EstimationJob:
    @dataclass(frozen=True)
    class Result:
        pass

    def __init__(
        self,
        space_x_fit_id: str,
        space_y_fit_id: str,
        correspondence_fit_id: str,
        estimator: DictConfig,
    ) -> None:
        self.space_x_fit_id = space_x_fit_id
        self.space_y_fit_id = space_y_fit_id
        self.correspondence_fit_id = correspondence_fit_id
        self.estimator = estimator

        self.result: Optional[EstimationJob.Result] = None

    def run(self) -> EstimationJob.Result:
        pass

    def __hash__(self) -> str:
        return hash(
            (
                self.space_x_fit_id,
                self.space_y_fit_id,
                self.correspondence_fit_id,
                self.estimator,
            )
        )


class TransformationJob:
    @dataclass(frozen=True)
    class Result:
        pass

    def __init__(
        self,
        estimator: EstimationJob,
        space_x_test_id: str,
        space_y_test_id: str,
    ) -> None:
        self.estimator = estimator
        self.space_x_test_id = space_x_test_id
        self.space_y_test_id = space_y_test_id

        self.result: Optional[TransformationJob.Result] = None

    def run(self) -> TransformationJob.Result:
        pass

    def __hash__(self) -> str:
        return hash((self.estimator, self.space_x_test_id, self.space_y_test_id))


class LatentJob:
    @dataclass(frozen=True)
    class Result:
        pass

    def __init__(
        self,
        correspondence_test_id: str,
        metric_id: str,
        transformation_job: TransformationJob,
    ) -> None:
        self.correspondence_test_id = correspondence_test_id
        self.metric_id = metric_id
        self.transformation_job = transformation_job

        self.result: Optional[LatentJob.Result] = None

    def run(self) -> LatentJob.Result:
        pass

    def __hash__(self) -> str:
        return hash((self.correspondence_test_id, self.metric_id, self.transformation_job))


class DownstreamJob:
    @dataclass(frozen=True)
    class Result:
        pass

    def __init__(
        self,
        correspondence_test_id: str,
        metric_id: str,
        transformation_job: TransformationJob,
        y_gt_test: Tuple[str, str, str],  # (dataset_id, split_id, label_name)
        decoder_y_fit_id: str,
    ) -> None:
        self.correspondence_test_id = correspondence_test_id
        self.metric_id = metric_id
        self.transformation_job = transformation_job
        self.y_gt_test = y_gt_test
        self.decoder_y_fit_id = decoder_y_fit_id

        self.result: Optional[DownstreamJob.Result] = None

    def run(self) -> DownstreamJob.Result:
        pass

    def __hash__(self) -> str:
        return hash(
            (
                self.correspondence_test_id,
                self.metric_id,
                self.transformation_job,
                self.y_gt_test,
                self.decoder_y_fit_id,
            )
        )


class AgreementJob:
    @dataclass(frozen=True)
    class Result:
        pass

    def __init__(
        self,
        correspondence_test_id: str,
        metric_id: str,
        transformation_job: TransformationJob,
        space_y_test_id: str,
        decoder_y_fit_id: str,
    ) -> None:
        self.correspondence_test_id = correspondence_test_id
        self.metric_id = metric_id
        self.transformation_job = transformation_job
        self.space_y_test_id = space_y_test_id
        self.decoder_y_fit_id = decoder_y_fit_id

        self.result: Optional[AgreementJob.Result] = None

    def run(self) -> AgreementJob.Result:
        pass

    def __hash__(self) -> str:
        return hash(
            (
                self.correspondence_test_id,
                self.metric_id,
                self.transformation_job,
                self.space_y_test_id,
                self.decoder_y_fit_id,
            )
        )


def compile_experiments(benchmark_name: str) -> str:
    with (BENCHMARK_DIR / benchmark_name / "benchmark.json").open("r") as f:
        experiments = json.load(f)

    print(experiments)
    # estimation_stage: Set[EstimationJob] = set()
    # transformation_stage: Set[TransformationJob] = set()
    # latent_stage: Set[LatentJob] = set()
    # downstream_stage: Set[DownstreamJob] = set()
    # agreement_stage: Set[AgreementJob] = set()


if __name__ == "__main__":
    benchmark_name = "eval_estimator"
    experiments = compile_experiments(benchmark_name)
