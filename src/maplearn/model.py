import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig


def make_model(grid, seed=42, d_model=128, max_moves=12, device="cpu"):
    if d_model % 4:
        raise ValueError("d_model must be divisible by four")
    cfg = HookedTransformerConfig(
        n_layers=2,
        n_heads=4,
        d_model=d_model,
        d_head=d_model // 4,
        d_mlp=4 * d_model,
        n_ctx=max_moves + 1,
        d_vocab=grid.vocab_size,
        d_vocab_out=grid.size**2,
        act_fn="gelu",
        normalization_type="LN",
        seed=seed,
        device=device,
        default_prepend_bos=False,
    )
    return HookedTransformer(cfg, move_to_device=True)


def final_logits(model, tokens, positions):
    logits = model(tokens)
    return logits[torch.arange(len(tokens), device=tokens.device), positions]
