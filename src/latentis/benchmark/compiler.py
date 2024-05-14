from __future__ import annotations

import hashlib
import json
from abc import abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import networkx as nx
from omegaconf import DictConfig

from latentis.benchmark import BENCHMARK_DIR


class Job:
    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def properties(self):
        pass

    def item_id(self) -> int:
        s = json.dumps(self.properties(), default=lambda o: o.__dict__, sort_keys=True).encode(encoding="utf-8")
        return int(hashlib.sha1(s).hexdigest(), 16) % (10**8)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"


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

    def properties(self) -> Tuple:
        return (
            self.space_x_fit_id,
            self.space_y_fit_id,
            self.correspondence_fit_id,
            self.estimator,
        )

    def __repr__(self) -> str:
        return f"{self.estimator}(Xf: {self.space_x_fit_id[:3]}, Yf: {self.space_y_fit_id[:3]}, Cf: {self.correspondence_fit_id[:3]})"

    def node_info(self) -> Dict[str, str]:
        return {
            "space_x_fit_id": self.space_x_fit_id,
            "space_y_fit_id": self.space_y_fit_id,
            "correspondence_fit_id": self.correspondence_fit_id,
            "estimator": self.estimator,
        }


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

    def properties(self) -> Tuple:
        return self.estimator, self.space_x_test_id, self.space_y_test_id

    def __repr__(self) -> str:
        return f"T(Xt: {self.space_x_test_id[:3]}, Yt: {self.space_y_test_id[:3]})"

    def node_info(self) -> Dict[str, str]:
        return {
            "space_x_test_id": self.space_x_test_id,
            "space_y_test_id": self.space_y_test_id,
        }


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

    def properties(self) -> Tuple:
        return (
            self.correspondence_test_id,
            self.metric_id,
            self.transformation_job,
        )

    def __repr__(self) -> str:
        return f"{self.metric_id[:3]}(Ct: {self.correspondence_test_id[:3]})"

    def node_info(self) -> Dict[str, str]:
        return {
            "correspondence_test_id": self.correspondence_test_id,
            "metric_id": self.metric_id,
        }


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

    def properties(self) -> Tuple:
        return (
            self.correspondence_test_id,
            self.metric_id,
            self.transformation_job,
            self.y_gt_test,
            self.decoder_y_fit_id,
        )

    def __repr__(self) -> str:
        return f"{self.metric_id[:3]}(Ct: {self.correspondence_test_id[:3]}, Ygtt: {self.y_gt_test[0]}, Dyf: {self.decoder_y_fit_id[:3]})"

    def node_info(self):
        return {
            "correspondence_test_id": self.correspondence_test_id,
            "metric_id": self.metric_id,
            "y_gt_test": self.y_gt_test,
            "decoder_y_fit_id": self.decoder_y_fit_id,
        }


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

    def properties(self) -> Tuple:
        return (
            self.correspondence_test_id,
            self.metric_id,
            self.transformation_job,
            self.space_y_test_id,
            self.decoder_y_fit_id,
        )

    def __repr__(self) -> str:
        return f"{self.metric_id[:3]}(Ct: {self.correspondence_test_id[:3]}, Yt: {self.space_y_test_id[:3]}, Dyf: {self.decoder_y_fit_id[:3]})"

    def node_info(self):
        return {
            "correspondence_test_id": self.correspondence_test_id,
            "metric_id": self.metric_id,
            "space_y_test_id": self.space_y_test_id,
            "decoder_y_fit_id": self.decoder_y_fit_id,
        }


def _add_to_graph(graph: nx.Graph, job: Job) -> int:
    if job.item_id() in graph.nodes:
        job = graph.nodes[job.item_id()]["job"]
    else:
        graph.add_node(job.item_id(), job=job)
    return job.item_id()


def compile_experiments(benchmark_name: str) -> Dict[str, Union[nx.Graph, str]]:
    with (BENCHMARK_DIR / benchmark_name / "benchmark.json").open("r") as f:
        experiments = json.load(f)

    graph = nx.Graph()
    estimations = set()
    transformations = set()
    downstreams = set()
    agreements = set()
    latents = set()

    for experiment in experiments:
        estimation_job = _add_to_graph(
            graph=graph,
            job=EstimationJob(
                space_x_fit_id=experiment["fit_x_space"]["__id"],
                space_y_fit_id=experiment["fit_y_space"]["__id"],
                correspondence_fit_id=experiment["fit_correspondence"]["__id"],
                estimator=experiment["estimator"],
            ),
        )
        estimations.add(estimation_job)

        transformation_job = _add_to_graph(
            graph=graph,
            job=TransformationJob(
                estimator=estimation_job,
                space_x_test_id=experiment["test_x_space"]["__id"],
                space_y_test_id=experiment["test_y_space"]["__id"],
            ),
        )
        graph.add_edge(estimation_job, transformation_job)
        transformations.add(transformation_job)

        if experiment["metric_type"] == "latent":
            latent_job = _add_to_graph(
                graph=graph,
                job=LatentJob(
                    correspondence_test_id=experiment["test_correspondence"]["__id"],
                    metric_id=experiment["metric"],
                    transformation_job=transformation_job,
                ),
            )
            latents.add(latent_job)
            graph.add_edge(transformation_job, latent_job)

        elif experiment["metric_type"] == "downstream":
            if experiment["test_decoder_y"]:
                down_job = _add_to_graph(
                    graph=graph,
                    job=DownstreamJob(
                        correspondence_test_id=experiment["test_correspondence"]["__id"],
                        metric_id=experiment["metric"],
                        transformation_job=transformation_job,
                        y_gt_test=(
                            experiment["y_test_gt"]["dataset"],
                            experiment["y_test_gt"]["split"],
                            experiment["y_test_gt"]["y_label"],
                        ),
                        decoder_y_fit_id=experiment["fit_y_space"]["__id"],
                    ),
                )
                downstreams.add(down_job)
                graph.add_edge(transformation_job, down_job)

        elif experiment["metric_type"] == "agreement":
            if experiment["test_decoder_y"]:
                agreement_job = _add_to_graph(
                    graph=graph,
                    job=AgreementJob(
                        correspondence_test_id=experiment["test_correspondence"]["__id"],
                        metric_id=experiment["metric"],
                        transformation_job=transformation_job,
                        space_y_test_id=experiment["test_y_space"]["__id"],
                        decoder_y_fit_id=experiment["fit_y_space"]["__id"],
                    ),
                )
                agreements.add(agreement_job)
                graph.add_edge(transformation_job, agreement_job)

        else:
            raise ValueError(f'Unknown metric type: {experiment["metric_type"]}')

    return graph, {
        "estimations": estimations,
        "transformations": transformations,
        "latents": latents,
        "downstreams": downstreams,
        "agreements": agreements,
    }


if __name__ == "__main__":
    benchmark_name = "eval_estimator"
    graph, _ = compile_experiments(benchmark_name)
    print(graph)
