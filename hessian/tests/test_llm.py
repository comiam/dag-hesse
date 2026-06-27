"""Toy SegmentedTransformer: attention-visible partition, LM loss, and Phi conformance.

White-box checks that the exp9 toy decoder exposes `get_param_groups` as a clean partition
with the query / key / value / output projections kept as separate (visible) blocks, that
the causal LM loss shifts correctly, and that the model plugs into `ParamBlockEstimator`
as a `ParamGroupedModel` (the seam exp9 measures). Numerical fidelity of Phi itself is
covered in `test_param_space`; here we verify structure and the integration seam.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments.llm import (  # noqa: E402
    CausalLMCrossEntropy,
    SegmentedTransformer,
    random_token_batch,
    sample_token_batch,
)
from hessian.param_space import ParamBlockEstimator  # noqa: E402


def _assert_partition(model: SegmentedTransformer) -> None:
    groups = model.get_param_groups()
    grouped = [p for ps in groups.values() for p in ps]
    grouped_ids = {id(p) for p in grouped}
    all_ids = {id(p) for p in model.parameters()}
    assert len(grouped_ids) == len(grouped), "a parameter appears in two groups"
    assert grouped_ids == all_ids, "groups must partition the model parameters exactly"


def test_param_groups_visible_attention() -> None:
    model = SegmentedTransformer.from_preset("tiny")  # n_layers = 2
    _assert_partition(model)

    names = list(model.get_param_groups().keys())
    expected = [
        "embed",
        "L0.attn_q",
        "L0.attn_k",
        "L0.attn_v",
        "L0.attn_o",
        "L0.mlp",
        "L1.attn_q",
        "L1.attn_k",
        "L1.attn_v",
        "L1.attn_o",
        "L1.mlp",
        "head",
    ]
    assert names == expected, f"group order / visibility wrong: {names}"

    # The pre-attention norm rides with the query block (BN-with-segment convention).
    groups = model.get_param_groups()
    ln1_ids = {
        id(p) for n, p in model.named_parameters() if n.startswith("blocks.0.ln1.")
    }
    q_ids = {id(p) for p in groups["L0.attn_q"]}
    assert ln1_ids <= q_ids, "pre-attention LayerNorm must ride with the query block"


def test_param_groups_block_granularity() -> None:
    model = SegmentedTransformer.from_preset("tiny", attn_granularity="block")
    _assert_partition(model)
    names = list(model.get_param_groups().keys())
    assert names == [
        "embed",
        "L0.attn",
        "L0.mlp",
        "L1.attn",
        "L1.mlp",
        "head",
    ], f"coarse attention grouping wrong: {names}"


def test_forward_shape() -> None:
    model = SegmentedTransformer.from_preset("tiny")
    ids = torch.randint(0, model.vocab_size, (2, 8))
    out = model(ids)
    assert out.shape == (2, 8, model.vocab_size), "logits shape (B, S, V)"


def test_from_preset_unknown_raises() -> None:
    raised = False
    try:
        SegmentedTransformer.from_preset("gpt5")
    except ValueError:
        raised = True
    assert raised, "unknown preset must raise ValueError"


def test_invalid_granularity_raises() -> None:
    raised = False
    try:
        SegmentedTransformer.from_preset("tiny", attn_granularity="heads")
    except ValueError:
        raised = True
    assert raised, "invalid attn_granularity must raise ValueError"


def test_causal_lm_cross_entropy_shift() -> None:
    torch.manual_seed(0)
    logits = torch.randn(2, 5, 7)
    targets = torch.randint(0, 7, (2, 5))
    loss = CausalLMCrossEntropy()(logits, targets)
    manual = F.cross_entropy(
        logits[:, :-1, :].reshape(-1, 7), targets[:, 1:].reshape(-1)
    )
    assert torch.allclose(loss, manual), "loss must score logits_t against token_{t+1}"


def test_random_token_batch_deterministic() -> None:
    a = random_token_batch(50, 4, 6, generator=torch.Generator().manual_seed(3))
    b = random_token_batch(50, 4, 6, generator=torch.Generator().manual_seed(3))
    assert a.shape == (4, 6) and a.dtype == torch.long, "shape / dtype"
    assert torch.equal(a, b), "same seed must give the same batch"
    assert int(a.min()) >= 0 and int(a.max()) < 50, "ids must lie in the vocabulary"


def test_sample_token_batch_is_padding_free() -> None:
    class _FakeTok:
        def __call__(self, text: str, return_tensors: str = "pt") -> SimpleNamespace:
            ids = torch.arange(len(text.split())).unsqueeze(0)
            return SimpleNamespace(input_ids=ids)

    batch = sample_token_batch(_FakeTok(), batch_size=3, seq_len=4)
    assert batch.shape == (3, 4), "concatenate-and-chunk must fill (batch, seq_len)"


def test_estimate_phi_report_respects_coupling_pairs() -> None:
    torch.manual_seed(0)
    model = SegmentedTransformer.from_preset("tiny")
    model.eval()
    ids = random_token_batch(
        model.vocab_size, 2, 8, generator=torch.Generator().manual_seed(0)
    )
    est = ParamBlockEstimator(model, CausalLMCrossEntropy(), n_probes=2)
    names = list(model.get_param_groups().keys())
    subset = [(names[0], names[1]), (names[1], names[2])]
    report = est.estimate_phi_report(ids, ids, coupling_pairs=subset)
    assert set(report.coupling) == set(subset), "coupling restricted to the given pairs"
    assert list(report.diag_frob.keys()) == names, "diagonals still cover every block"
    assert math.isfinite(report.phi), "phi is unaffected by the coupling subset"


def test_is_param_grouped_model_for_phi() -> None:
    torch.manual_seed(0)
    model = SegmentedTransformer.from_preset("tiny")
    model.eval()
    ids = random_token_batch(
        model.vocab_size, 2, 8, generator=torch.Generator().manual_seed(0)
    )
    report = ParamBlockEstimator(
        model, CausalLMCrossEntropy(), n_probes=2
    ).estimate_phi_report(ids, ids)

    names = list(model.get_param_groups().keys())
    assert list(report.diag_frob.keys()) == names, "diag_frob must be group-major"
    assert math.isfinite(report.phi), "phi must be finite"

    expected_pairs = {(v, w) for i, v in enumerate(names) for w in names[i + 1 :]}
    assert set(report.coupling) == expected_pairs, "coupling must cover all v<w pairs"


if __name__ == "__main__":
    test_param_groups_visible_attention()
    print("param_groups_visible_attention: OK")

    test_param_groups_block_granularity()
    print("param_groups_block_granularity: OK")

    test_forward_shape()
    print("forward_shape: OK")

    test_from_preset_unknown_raises()
    print("from_preset_unknown_raises: OK")

    test_invalid_granularity_raises()
    print("invalid_granularity_raises: OK")

    test_causal_lm_cross_entropy_shift()
    print("causal_lm_cross_entropy_shift: OK")

    test_random_token_batch_deterministic()
    print("random_token_batch_deterministic: OK")

    test_sample_token_batch_is_padding_free()
    print("sample_token_batch_is_padding_free: OK")

    test_estimate_phi_report_respects_coupling_pairs()
    print("estimate_phi_report_respects_coupling_pairs: OK")

    test_is_param_grouped_model_for_phi()
    print("is_param_grouped_model_for_phi: OK")

    print("test_llm: all checks passed")
