"""Validation and model-guided exact search."""

from dataclasses import dataclass

import torch

from .model import TinyRecursiveModel


@dataclass(frozen=True)
class SolveResult:
    grid: torch.Tensor | None
    nodes: int


def is_valid_solution(grid: torch.Tensor, clues: torch.Tensor | None = None) -> bool:
    board = grid.reshape(9, 9)
    target = set(range(1, 10))
    if clues is not None and not torch.equal(grid[clues > 0], clues[clues > 0]):
        return False
    for index in range(9):
        if set(board[index].tolist()) != target or set(board[:, index].tolist()) != target:
            return False
    for row in range(0, 9, 3):
        for column in range(0, 9, 3):
            if set(board[row : row + 3, column : column + 3].reshape(-1).tolist()) != target:
                return False
    return True


def candidates(board: torch.Tensor, position: int) -> list[int]:
    row, column = divmod(position, 9)
    used = set(board[row * 9 : row * 9 + 9].tolist())
    used.update(board[column::9].tolist())
    top, left = row // 3 * 3, column // 3 * 3
    used.update(board.reshape(9, 9)[top : top + 3, left : left + 3].reshape(-1).tolist())
    return [digit for digit in range(1, 10) if digit not in used]


@torch.no_grad()
def solve_guided(
    puzzle: torch.Tensor, model: TinyRecursiveModel | None = None, max_nodes: int = 100_000
) -> SolveResult:
    board = puzzle.clone()
    nodes = 0

    def visit() -> bool:
        nonlocal nodes
        nodes += 1
        if nodes > max_nodes:
            return False
        empty = (board == 0).nonzero(as_tuple=False).flatten().tolist()
        if not empty:
            return True
        options = [(len(candidates(board, position)), position) for position in empty]
        size, position = min(options)
        if size == 0:
            return False
        digits = candidates(board, position)
        if model is not None:
            scores = model(board)[0, position]
            digits.sort(key=lambda digit: float(scores[digit - 1]), reverse=True)
        for digit in digits:
            board[position] = digit
            if visit():
                return True
        board[position] = 0
        return False

    solved = visit()
    return SolveResult(board.clone() if solved else None, nodes)
