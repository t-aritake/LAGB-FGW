# LAGB-FGW: Label-Aware Graph-Based Fused Gromov-Wasserstein for Domain Adaptation

## Overview

This repository implements **LAGB-FGW**, a domain adaptation method that transfers labels from a labeled source domain to an unlabeled target domain using Optimal Transport (OT). The method extends Fused Gromov-Wasserstein (FGW) transport by incorporating two complementary regularizations:

- **LA (Label-Aware)**: Augments the source intra-domain cost with label similarity, encouraging samples of the same class to be transported together.
- **GB (Graph-Based)**: Replaces the Euclidean intra-domain cost on the target with shortest-path distances over a k-NN graph, capturing the intrinsic manifold structure of unlabeled target data.

## Problem Setting

- **Source domain**: Labeled samples `(Xs, ys)` with features in a high-dimensional space.
- **Target domain**: Unlabeled samples `Xt` sharing a subset of common features with the source.
- **Goal**: Predict target labels `yt` using the estimated OT transport plan and barycentric mapping.

## Methods

| Name | alpha | beta | Target cost | Source cost |
|------|-------|------|-------------|-------------|
| OT | 0 | 0 | Euclidean | Euclidean |
| GWOT | 1 | 0 | Euclidean | Euclidean |
| FGW | α | 0 | Euclidean | Euclidean |
| LA-FGW | α | β | Euclidean | Euclidean + label loss |
| GB-FGW | α | 0 | Graph (k-NN SP) | Euclidean |
| **LAGB-FGW** (ours) | α | β | Graph (k-NN SP) | Euclidean + label loss |

The FGW objective balances feature alignment (cross-domain cost `M`) and structural alignment (intra-domain costs `C1`, `C2`):

```
FGW(alpha) = alpha * GW(C1, C2) + (1 - alpha) * W(M)
```

For LA variants, `C1` is replaced by `(1 - beta) * C1 + beta * L` where `L` is the label cost matrix.

A simple baseline (`ClassificationAndMajorityVote`) uses OT-based label transfer followed by Spectral Clustering with majority vote assignment.

## Datasets

**Synthetic:**
- `linear` — Gaussian blobs with covariate shift
- `two_circles` — Two concentric circular distributions
- `two_moons` — Two interleaving half-moons
- `two_spirals` — Two interleaving spirals

**Real-world:**
- `har70_artificial_shift` — HAR70 human activity recognition dataset with artificially induced domain shift

## Repository Structure

```
run.py           # Main entry point (Hydra config, MLflow logging)
models.py        # GW, FGW, LAGB-FGW model classes
datasets.py      # Dataset generators (synthetic + HAR70)
utils.py         # LabelConverter, plotting utilities
boxplot.py       # Result visualization for synthetic datasets
boxplot_har70.py # Result visualization for HAR70
plot_appendix.py # Appendix figures
config.yaml      # Hydra configuration (dataset, model, hyperparameters)
```

## Usage

Experiments are configured via [Hydra](https://hydra.cc/) and tracked with [MLflow](https://mlflow.org/).

Start the MLflow tracking server:
```bash
mlflow server --port 50000
```

Run an experiment:
```bash
python run.py dataset=two_moons model=LAGB-FGW alpha=0.5 beta=0.3 n_neighbors=5
```

Key config parameters:

| Parameter | Description |
|-----------|-------------|
| `dataset` | Dataset name (`linear`, `two_circles`, `two_moons`, `two_spirals`, `har70_artificial_shift`) |
| `model` | Method name (`OT`, `GWOT`, `FGW`, `LA-FGW`, `GB-FGW`, `LAGB-FGW`, `simple_baseline`) |
| `alpha` | GW vs. feature distance trade-off (0 = pure OT, 1 = pure GW) |
| `beta` | Label loss weight in source cost |
| `n_neighbors` | Number of neighbors for k-NN graph (GB variants) |
| `seed` | Random seed |
| `make_plot` | Whether to log scatter plots to MLflow |

## Dependencies

- `POT` (Python Optimal Transport)
- `numpy`, `scipy`, `scikit-learn`
- `networkx`
- `hydra-core`, `omegaconf`
- `mlflow`
- `matplotlib`, `seaborn`, `pandas`
