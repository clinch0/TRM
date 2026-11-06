"""Evaluate neural predictions and guided exact solving on fixed 9x9 puzzles."""

from time import perf_counter

import torch

from .puzzles import PUZZLES, parse_grid
from .solver import is_valid_solution, solve_guided
from .train import train_model


def main() -> None:
    model, report = train_model()
    print(f"training loss={report.loss:.4f}, blank-cell accuracy={report.cell_accuracy:.3f}")
    print("name         neural cells  neural exact  guided exact  nodes  ms")
    for item in PUZZLES:
        puzzle, target = parse_grid(item.grid), parse_grid(item.solution)
        neural = model.predict(puzzle)[0]
        neural_cells = float((neural == target).float().mean())
        started = perf_counter()
        result = solve_guided(puzzle, model)
        elapsed_ms = (perf_counter() - started) * 1000
        neural_exact = is_valid_solution(neural, puzzle) and torch.equal(neural, target)
        guided_exact = result.grid is not None and is_valid_solution(result.grid, puzzle)
        print(
            f"{item.name:<12} {neural_cells:>11.1%}  {str(neural_exact):>12}  "
            f"{str(guided_exact):>12}  {result.nodes:>5}  {elapsed_ms:>5.1f}"
        )


if __name__ == "__main__":
    main()
