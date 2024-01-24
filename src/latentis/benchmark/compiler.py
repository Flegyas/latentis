from __future__ import annotations

import json
from abc import abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

from omegaconf import DictConfig

from latentis.benchmark import BENCHMARK_DIR


class Job:
    def __init__(self) -> None:
        self.children: Sequence[Job] = []

    def add_child(self, child: Job) -> None:
        self.children.append(child)

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def _get_hash_properties(self):
        pass

    def __hash__(self) -> str:
        return hash(self._get_hash_properties())

    # def __repr__(self) -> str:
    #     f = lambda f: f[:5] if isinstance(f, str) else f
    #     return f"{self.__class__.__name__}({json.dumps(list(map(f, self._get_hash_properties())), default=lambda o: list(map(f, o._get_hash_properties())))})"


class EstimationJob(Job):
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
        super().__init__()
        self.space_x_fit_id = space_x_fit_id
        self.space_y_fit_id = space_y_fit_id
        self.correspondence_fit_id = correspondence_fit_id
        self.estimator = estimator

        self.result: Optional[EstimationJob.Result] = None

    def run(self) -> EstimationJob.Result:
        pass

    def _get_hash_properties(self) -> Tuple:
        return (
            self.space_x_fit_id,
            self.space_y_fit_id,
            self.correspondence_fit_id,
            self.estimator,
        )


class TransformationJob(Job):
    @dataclass(frozen=True)
    class Result:
        pass

    def __init__(
        self,
        estimator: EstimationJob,
        space_x_test_id: str,
        space_y_test_id: str,
    ) -> None:
        super().__init__()
        self.estimator = estimator
        self.space_x_test_id = space_x_test_id
        self.space_y_test_id = space_y_test_id

        self.result: Optional[TransformationJob.Result] = None

    def run(self) -> TransformationJob.Result:
        pass

    def _get_hash_properties(self) -> Tuple:
        return self.estimator, self.space_x_test_id, self.space_y_test_id


class LatentJob(Job):
    @dataclass(frozen=True)
    class Result:
        pass

    def __init__(
        self,
        correspondence_test_id: str,
        metric_id: str,
        transformation_job: TransformationJob,
    ) -> None:
        super().__init__()
        self.correspondence_test_id = correspondence_test_id
        self.metric_id = metric_id
        self.transformation_job = transformation_job

        self.result: Optional[LatentJob.Result] = None

    def run(self) -> LatentJob.Result:
        pass

    def _get_hash_properties(self) -> Tuple:
        return (
            self.correspondence_test_id,
            self.metric_id,
            self.transformation_job,
        )


class DownstreamJob(Job):
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
        super().__init__()
        self.correspondence_test_id = correspondence_test_id
        self.metric_id = metric_id
        self.transformation_job = transformation_job
        self.y_gt_test = y_gt_test
        self.decoder_y_fit_id = decoder_y_fit_id

        self.result: Optional[DownstreamJob.Result] = None

    def run(self) -> DownstreamJob.Result:
        pass

    def _get_hash_properties(self) -> Tuple:
        return (
            self.correspondence_test_id,
            self.metric_id,
            self.transformation_job,
            self.y_gt_test,
            self.decoder_y_fit_id,
        )


class AgreementJob(Job):
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
        super().__init__()
        self.correspondence_test_id = correspondence_test_id
        self.metric_id = metric_id
        self.transformation_job = transformation_job
        self.space_y_test_id = space_y_test_id
        self.decoder_y_fit_id = decoder_y_fit_id

        self.result: Optional[AgreementJob.Result] = None

    def run(self) -> AgreementJob.Result:
        pass

    def _get_hash_properties(self) -> Tuple:
        return (
            self.correspondence_test_id,
            self.metric_id,
            self.transformation_job,
            self.space_y_test_id,
            self.decoder_y_fit_id,
        )


def _add_to_stage(stage: Dict[str, Job], job: Job) -> Job:
    if hash(job) not in stage:
        stage[hash(job)] = job
    return stage[hash(job)]


def compile_experiments(benchmark_name: str) -> Sequence[Job]:
    with (BENCHMARK_DIR / benchmark_name / "benchmark.json").open("r") as f:
        experiments = json.load(f)

    estimation_stage: Dict[str, EstimationJob] = {}
    transformation_stage: Dict[str, TransformationJob] = {}
    eval_stage: Dict[str, LatentJob] = {}

    for experiment in experiments:
        estimation_job = _add_to_stage(
            stage=estimation_stage,
            job=EstimationJob(
                space_x_fit_id=experiment["fit_x_space"]["__id"],
                space_y_fit_id=experiment["fit_y_space"]["__id"],
                correspondence_fit_id=experiment["fit_correspondence"]["__id"],
                estimator=experiment["estimator"],
            ),
        )

        transformation_job = _add_to_stage(
            stage=transformation_stage,
            job=TransformationJob(
                estimator=estimation_job,
                space_x_test_id=experiment["test_x_space"]["__id"],
                space_y_test_id=experiment["test_y_space"]["__id"],
            ),
        )
        estimation_job.add_child(transformation_job)

        if experiment["metric_type"] == "latent":
            eval_job = _add_to_stage(
                stage=eval_stage,
                job=LatentJob(
                    correspondence_test_id=experiment["test_correspondence"]["__id"],
                    metric_id=experiment["metric"],
                    transformation_job=transformation_job,
                ),
            )

        elif experiment["metric_type"] == "downstream":
            eval_job = _add_to_stage(
                stage=eval_stage,
                job=DownstreamJob(
                    correspondence_test_id=experiment["test_correspondence"]["__id"],
                    metric_id=experiment["metric"],
                    transformation_job=transformation_job,
                    y_gt_test=(
                        experiment["test_y_space"]["dataset"],
                        experiment["test_y_space"]["split"],
                        "y_label",  # experiment["test_y_space"]["label"],  # TODO: add this to the solver
                    ),
                    decoder_y_fit_id=experiment["fit_y_space"]["__id"],
                ),
            )
        elif experiment["metric_type"] == "agreement":
            eval_job = _add_to_stage(
                stage=eval_stage,
                job=AgreementJob(
                    correspondence_test_id=experiment["test_correspondence"]["__id"],
                    metric_id=experiment["metric"],
                    transformation_job=transformation_job,
                    space_y_test_id=experiment["test_y_space"]["__id"],
                    decoder_y_fit_id=experiment["fit_y_space"]["__id"],
                ),
            )
        else:
            raise ValueError(f'Unknown metric type: {experiment["metric_type"]}')

        transformation_job.add_child(eval_job)

    return estimation_stage.values()


def print_graph(job: Job, depth: int = 0) -> None:
    print("  " * depth + str(job))
    for child in job.children:
        print_graph(child, depth + 1)


if __name__ == "__main__":
    benchmark_name = "eval_estimator"
    graph = compile_experiments(benchmark_name)
    for job in graph:
        print_graph(job)
        print()
