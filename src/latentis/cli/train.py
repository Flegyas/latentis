from pathlib import Path
from typing import Annotated

import typer
from typer import Argument, Option

from latentis.benchmark.visualize import display_benchmark_graph
from latentis.data.train import attach_decoder

app = typer.Typer()


@app.command()
def decoder(
    config_path: Annotated[str, Option(help="Path to the config file")],
):
    attach_decoder()


if __name__ == "__main__":
    app()
