# database
from typing import Annotated

import typer
from typer import Argument

from latentis.benchmark.visualize import display_benchmark_graph

app = typer.Typer()

if __name__ == "__main__":
    app()
