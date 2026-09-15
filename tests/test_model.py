import torch

from maplearn.data import Grid, batch, generate
from maplearn.model import final_logits, make_model


def test_causality_padding_and_learning():
    torch.set_num_threads(2)
    grid = Grid()
    model = make_model(grid, d_model=32)
    rows = generate(grid, 8, "train", 4)
    tokens, pos, target = batch(rows, grid)
    model.eval()
    with torch.no_grad():
        logits = final_logits(model, tokens, pos)
        for i, row in enumerate(rows):
            t, p, _ = batch([row], grid)
            torch.testing.assert_close(
                logits[i], final_logits(model, t, p)[0], atol=1e-5, rtol=1e-5
            )
        first = torch.nn.functional.cross_entropy(logits, target).item()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    for _ in range(35):
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(
            final_logits(model, tokens, pos), target
        )
        loss.backward()
        optimizer.step()
    assert loss.item() < first * 0.2
    _, cache = model.run_with_cache(tokens, names_filter=["blocks.1.hook_resid_pre"])
    assert cache["blocks.1.hook_resid_pre"].shape == (*tokens.shape, 32)
