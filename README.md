# Latentis

<img align="center" alt="CI" src="https://img.shields.io/github/workflow/status/flegyas/latentis/Test%20Suite/main?label=main%20checks">
<img align="center" alt="Docs" src="https://img.shields.io/github/deployments/flegyas/latentis/github-pages?label=docs">
<img align="center" alt="NN Template" src="https://shields.io/badge/nn--template-0.2.3-emerald?style=flat&amp;labelColor=gray">
<img align="center" alt="Python" src="https://img.shields.io/badge/python-3.10-blue.svg">
<img align="center" alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg">

[![codecov](https://codecov.io/gh/Flegyas/latentis/graph/badge.svg?token=UQHBAEEUTM)](https://codecov.io/gh/Flegyas/latentis)

A Python package for analyzing and transforming neural latent spaces.


## Installation

```bash
pip install git+ssh://git@github.com/flegyas/latentis.git
```


## Quickstart

[comment]: <> (> Fill me!)


## Development installation

Setup the development environment:

```bash
git clone git@github.com:flegyas/latentis.git
cd latentis
conda env create -f env.yaml
conda activate latentis
pre-commit install
```

Run the tests:

```bash
pre-commit run --all-files
pytest -v
```


### Update the dependencies

Re-install the project in edit mode:

```bash
pip install -e .[dev]
```
