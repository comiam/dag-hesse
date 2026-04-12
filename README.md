# DAG-Hesse

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![CI](https://github.com/comiam/dag-hesse/actions/workflows/ci.yml/badge.svg)](https://github.com/comiam/dag-hesse/actions/workflows/ci.yml)

Code and experimental results for the paper:

> **Inter-Layer Hessian Analysis of Neural Networks with DAG Architectures**
> Maxim Bolshim, Alexander Kugaevskikh
> _Neural Networks_ (submitted, 2026)

## Overview

This repository provides an analytical framework for decomposing inter-layer
Hessian blocks in neural networks with arbitrary DAG architectures. The key
contributions include:

- Canonical decomposition $H = H^{GN} + H^T$ separating Gauss-Newton from
  tensor components for DAG-structured networks.
- Diagnostic metrics: **resonance** (R), **coupling** (C), **stable rank**,
  and **GN-Gap** for characterizing Hessian structure.
- Empirical validation across MLPs, ResNets, and convolutional architectures
  (ResNet-18 on CIFAR-10, ~11M parameters).

## Repository Structure

```
dag-hesse/
├── hessian/           # Core library: exact & stochastic Hessian computation
├── experiments/       # Experiment runners and configs
├── paper_data/        # Generate LaTeX tables & figures from results
├── results/           # Precomputed experimental results (JSON)
├── main.py            # CLI entry point
├── run_all.sh         # Run all experiments (tmux + multi-GPU)
├── Makefile           # Dev commands and experiment shortcuts
├── pyproject.toml     # Project metadata and dependencies
└── CITATION.cff       # Citation metadata
```

## Installation

Requires Python 3.10+ and PyTorch 2.0+.
A CUDA-capable GPU (e.g. NVIDIA RTX 4090) is recommended for running
experiments.

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

Datasets (CIFAR-10, CIFAR-100) are downloaded automatically by
`torchvision` on first run.

## Running Experiments

### Individual experiments

```bash
# GPU selection with GPU=N
make exp1
make exp6 GPU=2
```

### All experiments

```bash
# Distribute across available GPUs via tmux
./run_all.sh

# Or specific experiment on specific GPU
./run_all.sh exp1 --gpu 0
```

### Available experiments

| Command | Description                                                          |
| ------- | -------------------------------------------------------------------- |
| `exp1`  | Plain vs ResNet: decay of R and C with distance                      |
| `exp1b` | Spectral normalization verification (Theorem 6)                      |
| `exp2`  | Bottleneck ablation (CIFAR-100): sensitivity of R/C to narrow layers |
| `exp3`  | ReLU vs GELU: GN-Gap selectivity across activations                  |
| `exp4`  | Diamond MLP: tensor term $T_{u;v,w}$ activation                      |
| `exp5`  | Toy-Attention vs ReLU-MLP: $H^T_{Q,K} \neq 0$ verification           |
| `exp6`  | ResNet-18 on CIFAR-10: GN-Gap and R/C decay                          |

Results are saved as JSON files in the `results/` directory. The repository
ships precomputed results for all experiments, so tables and figures can be
generated without re-running.

Additional appendix variants (`exp1_appendix`, `exp2_unified`, `exp3_appendix`,
`exp3_ln`) are available; run `make help` for the full list.

## Generating Tables and Figures

Reproduce all LaTeX tables and TikZ figures from the paper:

```bash
python -m paper_data.main --results-dir results --output-dir output
```

## Development

```bash
make install    # Install with dev dependencies
make lint       # Run ruff + flake8
make format     # Format with black + ruff --fix
make typecheck  # Run mypy
make test       # Run pytest
make check      # Run lint + typecheck + test
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{bolshim2026interlayer,
  title   = {Inter-Layer Hessian Analysis of Neural Networks
             with {DAG} Architectures},
  author  = {Bolshim, Maxim and Kugaevskikh, Alexander},
  journal = {Neural Networks},
  year    = {2026},
  note    = {Submitted}
}
```

## License

This project is licensed under the [MIT License](LICENSE).
