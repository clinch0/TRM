"""Deterministic generation of valid solved grids and training puzzles."""

import random

import torch


def _pattern(row: int, column: int) -> int:
    return (row * 3 + row // 3 + column) % 9


def solved_grid(rng: random.Random) -> torch.Tensor:
    rows = [group * 3 + row for group in rng.sample(range(3), 3) for row in rng.sample(range(3), 3)]
    columns = [
        group * 3 + column
        for group in rng.sample(range(3), 3)
        for column in rng.sample(range(3), 3)
    ]
    digits = rng.sample(range(1, 10), 9)
    return torch.tensor([digits[_pattern(row, column)] for row in rows for column in columns])


def make_dataset(count: int, blanks: tuple[int, int] = (36, 58), seed: int = 7):
    rng = random.Random(seed)
    puzzles, solutions = [], []
    for _ in range(count):
        solution = solved_grid(rng)
        puzzle = solution.clone()
        empty = rng.sample(range(81), rng.randint(*blanks))
        puzzle[empty] = 0
        puzzles.append(puzzle)
        solutions.append(solution)
    return torch.stack(puzzles), torch.stack(solutions)
