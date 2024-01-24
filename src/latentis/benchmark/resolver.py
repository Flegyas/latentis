from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Optional

from omegaconf import OmegaConf

from latentis.benchmark import BENCHMARK_DIR
from latentis.correspondence import correspondences_index
from latentis.data import DATA_DIR
from latentis.data.dataset import LatentisDataset

BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)


def resolve_benchmark(benchmark_name: str) -> list[dict]:
    config = OmegaConf.load(BENCHMARK_DIR / benchmark_name / "benchmark_config.yaml")
    data_dir = Path(config.data_root) if config.data_root is not None else DATA_DIR

    experiments = []

    for pairing_policy in config.pairing_policy:
        train_split = pairing_policy.fit.split
        available_fit_correspondences = correspondences_index.get_items(
            split=train_split,
            **OmegaConf.merge(
                pairing_policy.correspondence_constraints,
                pairing_policy.fit.correspondence_constraints,
            ),
        )

        for fit_corr_id, fit_corr_properties in available_fit_correspondences.items():
            x_dataset_fit_name = fit_corr_properties["x_dataset"]
            y_dataset_fit_name = fit_corr_properties["y_dataset"]

            # TODO: how to access spaces properties without dataset loading?
            x_dataset_fit = LatentisDataset.load_from_disk(data_dir / x_dataset_fit_name, load_hf_dataset=False)
            y_dataset_fit = LatentisDataset.load_from_disk(data_dir / y_dataset_fit_name, load_hf_dataset=False)

            x_spaces_fit = x_dataset_fit.encodings.get_items(
                dataset=x_dataset_fit_name,
                split=train_split,
                **OmegaConf.merge(
                    pairing_policy.space_constraints,
                    pairing_policy.fit.x_space_constraints,
                ),
            )

            y_spaces_fit = y_dataset_fit.encodings.get_items(
                dataset=y_dataset_fit_name,
                split=train_split,
                **OmegaConf.merge(
                    pairing_policy.space_constraints,
                    pairing_policy.fit.y_space_constraints,
                ),
            )
            for x_space_fit_key, y_space_fit_key in itertools.product(x_spaces_fit, y_spaces_fit):
                for test_pairing_policy in pairing_policy.tests:
                    test_split = test_pairing_policy.split

                    available_test_correspondences = correspondences_index.get_items(
                        split=test_split,
                        **OmegaConf.merge(
                            pairing_policy.correspondence_constraints,
                            test_pairing_policy.correspondence_constraints,
                        ),
                    )
                    for test_corr_id, test_corr_properties in available_test_correspondences.items():
                        x_dataset_test_name = test_corr_properties["x_dataset"]
                        y_dataset_test_name = test_corr_properties["y_dataset"]

                        # TODO: how to access spaces properties without dataset loading?
                        x_dataset_test = LatentisDataset.load_from_disk(
                            data_dir / x_dataset_test_name, load_hf_dataset=False
                        )
                        y_dataset_test = LatentisDataset.load_from_disk(
                            data_dir / y_dataset_test_name, load_hf_dataset=False
                        )

                        x_spaces_test = x_dataset_test.encodings.get_items(
                            split=test_split,
                            dataset=x_dataset_test_name,
                            **OmegaConf.merge(
                                pairing_policy.space_constraints,
                                test_pairing_policy.x_space_constraints,
                            ),
                        )

                        y_spaces_test = y_dataset_test.encodings.get_items(
                            split=test_split,
                            dataset=y_dataset_test_name,
                            **OmegaConf.merge(
                                pairing_policy.space_constraints,
                                test_pairing_policy.y_space_constraints,
                            ),
                        )

                        for x_space_test_key, y_space_test_key in itertools.product(x_spaces_test, y_spaces_test):
                            for estimator_name in pairing_policy.estimators:
                                for metric_type, metrics in config.metrics.items():
                                    for metric_name in metrics:
                                        experiments.append(
                                            {
                                                "fit_correspondence": {
                                                    "__id": fit_corr_id,
                                                    **fit_corr_properties,
                                                },
                                                "fit_x_space": {
                                                    "__id": x_space_fit_key,
                                                    **x_spaces_fit[x_space_fit_key],
                                                },
                                                "fit_y_space": {
                                                    "__id": y_space_fit_key,
                                                    **y_spaces_fit[y_space_fit_key],
                                                },
                                                "test_correspondence": {
                                                    "__id": test_corr_id,
                                                    **test_corr_properties,
                                                },
                                                "test_x_space": {
                                                    "__id": x_space_test_key,
                                                    **x_spaces_test[x_space_test_key],
                                                },
                                                "test_y_space": {
                                                    "__id": y_space_test_key,
                                                    **y_spaces_test[y_space_test_key],
                                                },
                                                "estimator": estimator_name,
                                                "metric": metric_name,
                                                "metric_type": metric_type,
                                            }
                                        )

        (BENCHMARK_DIR / benchmark_name / "benchmark.json").write_text(f"{json.dumps(experiments, indent=4)}\n")
        return experiments


def experiments_summary(experiments: list[dict], benchmark_name: Optional[str] = None) -> str:
    fit_correspondences = set()
    fit_x_spaces = set()
    fit_y_spaces = set()
    test_correspondences = set()
    test_x_spaces = set()
    test_y_spaces = set()
    estimators = set()
    metrics = set()

    for experiment in experiments:
        fit_correspondences.add(repr(experiment["fit_correspondence"]))
        fit_x_spaces.add(repr(experiment["fit_x_space"]))
        fit_y_spaces.add(repr(experiment["fit_y_space"]))
        test_correspondences.add(repr(experiment["test_correspondence"]))
        test_x_spaces.add(repr(experiment["test_x_space"]))
        test_y_spaces.add(repr(experiment["test_y_space"]))
        estimators.add(repr(experiment["estimator"]))
        metrics.add((repr(experiment["metric"]), repr(experiment["metric_type"])))

    return f"""Benchmark {benchmark_name if benchmark_name is not None else '' } (n={len(experiments)}):
    Fit correspondences: {len(fit_correspondences)}
    Fit x spaces: {len(fit_x_spaces)}
    Fit y spaces: {len(fit_y_spaces)}
    Test correspondences: {len(test_correspondences)}
    Test x spaces: {len(test_x_spaces)}
    Test y spaces: {len(test_y_spaces)}
    Estimators: {len(estimators)}
    Metrics: {len(metrics)}
    """


def benchmark_summary(benchmark_name: str) -> str:
    with (BENCHMARK_DIR / benchmark_name / "benchmark.json").open("r") as f:
        experiments = json.load(f)
    return experiments_summary(experiments=experiments, benchmark_name=benchmark_name)


if __name__ == "__main__":
    benchmark_name = "eval_estimator"
    experiments = resolve_benchmark(benchmark_name)
    print(experiments_summary(experiments))
