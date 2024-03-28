from typing import Annotated

import typer
from typer import Argument

from latentis.benchmark.visualize import display_benchmark_graph

app = typer.Typer()

@app.command()
def a(name: Annotated[str, Argument(help="Name of the benchmark")]):
    print('a')


if __name__ == "__main__":
    app()
