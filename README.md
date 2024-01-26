
:construction::construction::construction: **Developers At Work. The library will be ready next month. Feel free to contact us directly for any specific request!** :construction::construction::construction:

---

# Latentis: Your Gateway to Latent Space Communication

<p align="center">
    <img align="center" alt="CI" src="https://github.com/Flegyas/latentis/actions/workflows/test_suite.yml/badge.svg?branch=main">
    <img align="center" alt="Coverage" src="https://codecov.io/gh/Flegyas/latentis/graph/badge.svg?token=UQHBAEEUTM"/>
    <img align="center" alt="Docs" src="https://img.shields.io/github/deployments/flegyas/latentis/github-pages?label=docs">
    <img align="center" alt="Python" src="https://img.shields.io/pypi/pyversions/latentis">
    <img align="center" alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg">
</p>

Welcome to **Latentis**, *the first-of-its-kind* Python library dedicated to the innovative field of [Latent Space Communication](https://github.com/UniReps/UniReps-resources). Latentis is designed to empower researchers, data scientists, and enthusiasts to unlock new insights by providing a comprehensive suite of tools where latent spaces are the core ingredient.


## Core Features

Latentis offers a structured suite of tools designed for efficiency and ease of use:
- **Data Download & Processing**: streamline the acquisition and preparation of complex datasets (via HuggingFace Datasets).
- **Advanced Encoding**: either employ pre-trained models or bring your own to encode anything.
- **Benchmarking Tools**: standard and customizable benchmarking tools, allowing for thorough evaluation and refinement of methods.

## Getting Started

Ease into your next research project with:
```bash
pip install latentis
```

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
