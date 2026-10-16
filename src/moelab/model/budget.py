"""Analytical unique-weight accounting; active counts are not measured FLOPs."""

from moelab.config import ModelConfig


def parameter_budget(cfg: ModelConfig) -> dict[str, int]:
    d, layers = cfg.d_model, cfg.n_layers
    embedding = cfg.vocab_size * d  # Tied output projection counted once.
    attention = layers * (2 * d * d + 2 * d * cfg.n_kv_heads * cfg.head_dim)
    norms = (2 * layers + 1) * d
    expert = 3 * d * cfg.effective_ffn_hidden_dim  # Three bias-free SwiGLU matrices.
    experts = cfg.moe.n_experts if cfg.moe else 1
    selected = cfg.moe.top_k if cfg.moe else 1
    router = layers * d * experts if cfg.moe else 0
    shared = embedding + attention + norms + router
    return {
        "ffn_hidden_dim": cfg.effective_ffn_hidden_dim,
        "embedding_params": embedding,
        "attention_params": attention,
        "norm_params": norms,
        "router_params": router,
        "ffn_params_per_expert_per_layer": expert,
        "total_ffn_params": layers * experts * expert,
        "active_ffn_params_estimate": layers * selected * expert,
        "total_params": shared + layers * experts * expert,
        "active_params_estimate": shared + layers * selected * expert,
    }


def compare_budgets(dense: ModelConfig, moe: ModelConfig) -> dict:
    """Require a shared backbone; report the signed MoE-minus-dense gap."""
    if dense.moe is not None or moe.moe is None:
        raise ValueError("Provide a dense model followed by an MoE model")
    fields = (
        "vocab_size",
        "d_model",
        "n_layers",
        "n_heads",
        "n_kv_heads",
        "max_seq_len",
        "dropout",
    )
    mismatches = [name for name in fields if getattr(dense, name) != getattr(moe, name)]
    if mismatches:
        raise ValueError(f"Comparison backbone mismatch: {', '.join(mismatches)}")
    a, b = parameter_budget(dense), parameter_budget(moe)
    gap = b["active_params_estimate"] - a["active_params_estimate"]
    fraction = gap / a["active_params_estimate"]
    return {
        "budget_basis": "shared weights plus selected experts per token; not FLOPs or memory",
        "dense": a,
        "moe": b,
        "active_gap_params": gap,
        "active_gap_fraction": fraction,
        "within_0_1_percent": abs(fraction) <= 0.001,
    }
