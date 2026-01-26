# Two-stage model for Genome-Wide Association Studies in Human Pathogens

This repository contains code for the two-stage model and supporting notebooks to reproduce the published findings.

Python 3.7 is required, as well as specific versions of packages, so please use a virtual environment.

```bash
pyenv local 3.7

python -m venv .venv

source .venv/bin/activate

pip install poetry

poetry install
```

## Repo structure
The `neural_net` folder contains code  used to generate the data and analyses described in the manuscript. Models are contained in `neural_net/models.py`
The `data_processing` folder contains scripts used to fetch and pre-process the data prior to the two-stage neural network model. They are meant to run on HPC environments.
The `manuscript_notebooks` were used to generate the figures used in the manuscript.


