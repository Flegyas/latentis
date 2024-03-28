from typing import Annotated, Optional

import typer
from typer import Argument, Option

from latentis.benchmark.resolver import experiments_summary, resolve_benchmark
from latentis.benchmark.visualize import display_benchmark_graph

app = typer.Typer()


@app.command()
def visualize(name: Annotated[str, Argument(help="Name of the benchmark")]):
    display_benchmark_graph(name)


@app.command()
def summary(name: Annotated[str, Argument(help="Name of the benchmark")]):
    experiments_summary(name)


@app.command()
def resolve(
    name: Annotated[str, Argument(help="Name of the benchmark")],
    overwrite: Annotated[Optional[bool], Option(help="Force re-resolve")] = False,
):
    resolve_benchmark(name)


@app.command()
def run(
    name: Annotated[str, Argument(help="Name of the benchmark")],
    overwrite: Annotated[Optional[bool], Option(help="Force re-run")] = False,
):
    print("run")


if __name__ == "__main__":
    app()
