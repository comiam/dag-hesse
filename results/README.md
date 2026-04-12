# Canonical experiment results

Single-source-of-truth for all tables and figures in the TMLR paper.

## Provenance

| Directory        | Experiment                            | Hessian mode            | Source                    |
|------------------|---------------------------------------|-------------------------|---------------------------|
| `exp1/`          | Plain vs ResNet (L=8,10,12)           | exact                   | `run_all.sh -> exp1`      |
| `exp1b/`         | Spectral-norm verification            | exact                   | `run_all.sh -> exp1b`     |
| `exp1_appendix/` | Deep nets (L=16,32, w=128)            | stochastic (100 probes) | `run_all.sh -> exp1_appx` |
| `exp2_unified/`  | Bottleneck ablation (L=6,8, base=512) | stochastic (100 probes) | `run_all.sh -> exp2_uni`  |
| `exp3/`          | GN-Gap by activation (no LN)          | exact                   | `run_all.sh -> exp3`      |
| `exp3_ln/`       | GN-Gap with LayerNorm                 | exact                   | `run_all.sh -> exp3_ln`   |
| `exp3_appendix/` | Exact vs stochastic ablation          | stochastic (100 probes) | `run_all.sh -> exp3_appx` |
| `exp4/`          | Diamond MLP (k=1,2,3)                 | exact                   | `run_all.sh -> exp4`      |
| `exp5/`          | Attention vs ReLU-MLP                 | exact                   | `run_all.sh -> exp5`      |
| `exp6/`          | ResNet-18 CIFAR-10 (ReLU, SiLU)       | stochastic (30 probes)  | `run_all.sh -> exp6`      |

## Usage

```bash
python -m paper_data.main \
  --results-dir results \
  --output-dir output
```
