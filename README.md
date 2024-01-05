# Latentis

<p align="center">
    <img align="center" alt="CI" src="https://github.com/Flegyas/latentis/actions/workflows/test_suite.yml/badge.svg?branch=main">
    <img align="center" alt="Coverage" src="https://codecov.io/gh/Flegyas/latentis/graph/badge.svg?token=UQHBAEEUTM"/>
    <img align="center" alt="Docs" src="https://img.shields.io/github/deployments/flegyas/latentis/github-pages?label=docs">
    <img align="center" alt="Python" src="https://img.shields.io/pypi/pyversions/latentis">
    <img align="center" alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg">
</p>


A Python package for analyzing and transforming neural latent spaces.


## Installation

```bash
pip install latentis
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
