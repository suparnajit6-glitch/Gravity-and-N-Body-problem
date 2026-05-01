# Gravity and N-Body Problem

Python experiments for gravitational N-body simulations, including a general
random-body simulation and an Alpha Centauri / Proxima Centauri example.

## Project Layout

```text
src/gravity_nbody/   Package source code
notebooks/           Exploratory Jupyter notebooks
environment.yml      Conda environment definition
```

## Setup

```bash
conda env create -f environment.yml
conda activate nbody_env
pip install -e .
```

## Run

```bash
python -m gravity_nbody.n_body_problem
python -m gravity_nbody.alpha_centauri
```

The Alpha Centauri data helpers use `astroquery`/SIMBAD when velocity data is
requested, so those calls require network access.
