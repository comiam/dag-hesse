"""Exp9: Stage-A "A3" frozen-transformer Phi ladder (the diagnosis, in language models).

Exp9 carries exp8's parameter-space blind-spot diagnostic onto decoder-only transformers:
with no training, for each frozen model it reports the discarded-coupling fraction

  Phi = 1 - sum_v ||H_{theta_v, theta_v}||_F^2 / ||H||_F^2  in [0, 1)   (Delta-M1)

- the curvature mass a block-diagonal preconditioner (K-FAC / EKFAC) throws away on a
transformer - together with the coupling field C(v, w) that locates it (Delta-M2). Both
come from one parameter-space backward graph via `ParamBlockEstimator.estimate_phi_report`.

Attention is kept *visible*: each layer's query / key / value / output projections are
separate blocks, so the coupling the paper isolates in attention (H^T_{Q,K} != 0, exp5)
is read off directly. The Stage-A "A3" question is how Phi climbs the model-size ladder
and whether pretraining (source (P)) inflates it.

Two backends share the runner (`Exp9Config`):

  - ``toy`` (the local, dependency-free ladder): `SegmentedTransformer` at growing sizes,
    measured on a random token minibatch. Self-contained - this is what the test-suite and
    the default sweep run.
  - ``hf`` (the on-demand ladder): frozen Hugging Face causal LMs (Qwen2.5, Llama-3.1)
    measured on a fixed text minibatch, loaded in bf16 with gradients enabled. The larger
    rungs need on-demand accelerator memory; ``transformers`` is imported lazily.

For each model the network is built (toy) or loaded (hf) once under a fixed build seed and
a fixed-size token minibatch is drawn; ``seeds`` then vary only the Hutchinson probes, so
Phi / the diagonal norms / the coupling field are measured once per seed and aggregated
into per-model mean/std plus the parameter count that indexes the ladder. The bundle is
JSON-serializable.
"""

from __future__ import annotations

import logging
from typing import Any, cast

import numpy as np
import torch
from torch import Tensor

from experiments.config import Exp9Config
from experiments.llm import (
    CausalLMCrossEntropy,
    SegmentedTransformer,
    load_hf_segmented,
    random_token_batch,
    resolve_dtype,
    sample_token_batch,
)
from experiments.utils import phi_report_to_dict, set_seed
from hessian import ParamBlockEstimator, ParamGroupedModel

logger = logging.getLogger(__name__)

# Fixed seed for model construction and the measurement batch, so a model's identity and
# its inputs are held constant while ``seeds`` vary only the Hutchinson probes.
_BUILD_SEED = 0


# ======================================================================
# Model / token-batch construction
# ======================================================================


def _build_model(
    cfg: Exp9Config,
    model_id: str,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[ParamGroupedModel, Any]:
    """(model, tokenizer) for ``model_id``; tokenizer is None on the toy backend.

    Gradients stay enabled (parameter-space Phi differentiates twice w.r.t. the weights);
    ``eval`` only disables stochastic layers - neither backend trains here.
    """
    if cfg.backend == "toy":
        toy = SegmentedTransformer.from_preset(
            model_id, attn_granularity=cfg.attn_granularity
        )
        toy.to(device=device, dtype=dtype)
        toy.eval()
        return toy, None
    if cfg.backend == "hf":
        hf, tokenizer = load_hf_segmented(
            model_id, dtype=cfg.dtype, attn_granularity=cfg.attn_granularity
        )
        hf.to(device)
        hf.eval()
        return hf, tokenizer
    raise ValueError(f"unknown backend {cfg.backend!r}; expected 'toy' or 'hf'")


def _make_token_batch(
    cfg: Exp9Config,
    model: ParamGroupedModel,
    tokenizer: Any,
    device: torch.device,
) -> Tensor:
    """A fixed ``(hessian_batch_size, seq_len)`` token minibatch for the measurement.

    Toy: a deterministic block of random ids over the model's vocabulary. HF: a
    padding-free chunk of the built-in sample text tokenized by the model's tokenizer.
    """
    if cfg.backend == "toy":
        gen = torch.Generator().manual_seed(_BUILD_SEED)
        vocab = cast(SegmentedTransformer, model).vocab_size
        ids = random_token_batch(
            vocab, cfg.hessian_batch_size, cfg.seq_len, generator=gen
        )
    else:
        ids = sample_token_batch(tokenizer, cfg.hessian_batch_size, cfg.seq_len)
    return ids.to(device)


def _count_params(model: ParamGroupedModel) -> int:
    """Total parameters across the measured blocks - the ladder's size axis."""
    return sum(
        p.numel() for params in model.get_param_groups().values() for p in params
    )


# ======================================================================
# Phi measurement
# ======================================================================


def _layer_key(name: str) -> str | None:
    """The layer tag of a group name (``"L3.attn_q" -> "L3"``); None for embed / head."""
    head = name.split(".", 1)[0]
    if head.startswith("L") and head[1:].isdigit():
        return head
    return None


def _coupling_pairs(names: list[str], band: int | None) -> list[tuple[str, str]] | None:
    """Block pairs whose coupling to measure: all (None) or a depth band + intra-layer.

    ``band is None`` returns None - the estimator then measures every v-before-w pair (the
    O(L^2) toy default). An int ``band`` keeps a pair (v before w in group order) when the
    two blocks are at most ``band`` apart in that order *or* share a transformer layer, so
    every intra-layer q/k/v/o/mlp coupling (the attention hotspots) survives while the
    cross-layer field is limited to a depth band - O(L) pairs for a many-block LLM.
    """
    if band is None:
        return None
    keys = [_layer_key(n) for n in names]
    pairs: list[tuple[str, str]] = []
    for i, v in enumerate(names):
        for j in range(i + 1, len(names)):
            same_layer = keys[i] is not None and keys[i] == keys[j]
            if same_layer or (j - i) <= band:
                pairs.append((v, names[j]))
    return pairs


def _measure_phi(
    model: ParamGroupedModel,
    input_ids: Tensor,
    loss_fn: torch.nn.Module,
    n_probes: int,
    coupling_band: int | None,
) -> dict[str, Any]:
    """JSON-safe Phi report on the token minibatch (x = y = input_ids; loss shifts).

    ``coupling_band`` scopes the coupling field (None = every pair; see `_coupling_pairs`);
    Phi and the diagonal norms always cover every block.
    """
    estimator = ParamBlockEstimator(model, loss_fn, n_probes=n_probes)
    names = list(model.get_param_groups().keys())
    pairs = _coupling_pairs(names, coupling_band)
    report = estimator.estimate_phi_report(input_ids, input_ids, coupling_pairs=pairs)
    return phi_report_to_dict(report)


# ======================================================================
# Aggregation
# ======================================================================


def _mean_std(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {"mean": float(arr.mean()), "std": float(arr.std())}


def _aggregate(
    raw: dict[str, dict[str, Any]],
    n_params: dict[str, int],
) -> dict[str, Any]:
    """Aggregates over seeds: per-model mean/std of Phi, diagonal norms, and couplings.

    Each model keeps its own block names (different depths give different keys), so models
    are summarized independently; the cross-model comparison is Phi vs ``n_params``.
    """
    out: dict[str, Any] = {}
    for model_id, per_seed in raw.items():
        reports = list(per_seed.values())
        diag_keys = reports[0]["diag_frob"].keys()
        coupling_keys = reports[0]["coupling"].keys()
        out[model_id] = {
            "phi": _mean_std([r["phi"] for r in reports]),
            "diag_frob": {
                k: _mean_std([r["diag_frob"][k] for r in reports]) for k in diag_keys
            },
            "coupling": {
                k: _mean_std([r["coupling"][k] for r in reports]) for k in coupling_keys
            },
            "n_params": n_params[model_id],
            "n_seeds": len(reports),
        }
    return out


# ======================================================================
# Runner
# ======================================================================


class Exp9Runner:
    """Runs the Stage-A frozen-transformer Phi ladder for one `Exp9Config`."""

    def __init__(self, config: Exp9Config, device: torch.device) -> None:
        self._config = config
        self._device = device

    def run(self) -> dict[str, Any]:
        cfg = self._config
        dtype = resolve_dtype(cfg.dtype)
        loss_fn = CausalLMCrossEntropy()

        raw: dict[str, dict[str, Any]] = {}
        n_params: dict[str, int] = {}
        for model_id in cfg.models:
            logger.info("exp9: %s (backend=%s)", model_id, cfg.backend)
            set_seed(_BUILD_SEED)
            model, tokenizer = _build_model(cfg, model_id, self._device, dtype)
            input_ids = _make_token_batch(cfg, model, tokenizer, self._device)
            n_params[model_id] = _count_params(model)

            per_seed: dict[str, Any] = {}
            for seed in cfg.seeds:
                set_seed(seed)
                report = _measure_phi(
                    model, input_ids, loss_fn, cfg.n_probes, cfg.coupling_band
                )
                per_seed[f"seed_{seed}"] = report
                logger.info("  seed %d: phi=%.4f", seed, report["phi"])
            raw[model_id] = per_seed

        return _aggregate(raw, n_params)
