"""One fixed random cell labeling per experiment; disjoint complete routes."""

import hashlib
import random
from dataclasses import dataclass

import torch

DIRECTIONS = ((0, -1), (1, 0), (0, 1), (-1, 0))  # N E S W


@dataclass(frozen=True)
class Grid:
    size: int = 8
    topology: str = "wrap"
    seed: int = 42

    def __post_init__(self):
        if self.size < 2 or self.topology not in ("wrap", "bounded", "portal"):
            raise ValueError("size must be >= 2; topology must be wrap, bounded or portal")
        if self.topology == "portal" and self.size < 4:
            raise ValueError("Portal worlds require size >= 4")

    @property
    def labels(self):
        labels = list(range(self.size**2))
        random.Random(self.seed).shuffle(labels)
        return labels

    @property
    def vocab_size(self):
        return self.size**2 + 5  # cells, four moves, padding

    def step(self, cell, direction):
        if not 0 <= cell < self.size**2 or not 0 <= direction < 4:
            raise ValueError("Invalid cell or direction")
        if self.topology == "portal":
            # Rewire two east/west edges, preserving reverse moves and degree.
            # The two origins are separated by half the grid along both axes.
            a, b = 0, (self.size // 2) * (self.size + 1)
            rewired = {(a, 1): b + 1, (b + 1, 3): a,
                       (b, 1): a + 1, (a + 1, 3): b}
            if (cell, direction) in rewired:
                return rewired[cell, direction]
        x, y = cell % self.size, cell // self.size
        dx, dy = DIRECTIONS[direction]
        x, y = x + dx, y + dy
        if self.topology in ("wrap", "portal"):
            x, y = x % self.size, y % self.size
        elif not (0 <= x < self.size and 0 <= y < self.size):
            raise ValueError("Move leaves bounded grid")
        return y * self.size + x


def partition(key):
    value = (
        int.from_bytes(hashlib.blake2b(bytes(key), digest_size=8).digest(), "little")
        % 100
    )
    return "train" if value < 80 else "val" if value < 90 else "test"


def generate(grid, count, split, seed, min_moves=2, max_moves=12):
    if (
        split not in ("train", "val", "test")
        or count < 1
        or not 1 <= min_moves <= max_moves
    ):
        raise ValueError("Invalid route generation parameters")
    if grid.size**2 > 250:
        raise ValueError("Route hashing supports grids with at most 250 cells")
    rng = random.Random(seed)
    seen, rows = set(), []
    labels = grid.labels
    for _ in range(max(10000, count * 200)):
        start = rng.randrange(grid.size**2)
        cell, moves = start, []
        for _ in range(rng.randint(min_moves, max_moves)):
            valid = []
            for direction in range(4):
                try:
                    valid.append((direction, grid.step(cell, direction)))
                except ValueError:
                    pass
            direction, cell = rng.choice(valid)
            moves.append(direction)
        key = (start, *moves)
        if key in seen or partition(key) != split:
            continue
        seen.add(key)
        # Input is only the initial cell and moves: no intermediate answer leakage.
        tokens = [labels[start]] + [grid.size**2 + move for move in moves]
        rows.append(
            {"tokens": tokens, "target": labels[cell], "cell": cell, "key": key}
        )
        if len(rows) == count:
            return rows
    raise ValueError("Requested too many unique routes; widen the length range")


def batch(rows, grid, device="cpu"):
    lengths = torch.tensor([len(row["tokens"]) for row in rows], device=device)
    tokens = torch.full(
        (len(rows), int(lengths.max())),
        grid.vocab_size - 1,
        dtype=torch.long,
        device=device,
    )
    for i, row in enumerate(rows):
        tokens[i, : len(row["tokens"])] = torch.tensor(row["tokens"], device=device)
    targets = torch.tensor([row["target"] for row in rows], device=device)
    return tokens, lengths - 1, targets
