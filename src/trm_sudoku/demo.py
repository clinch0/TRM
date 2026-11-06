"""Train a tiny model and solve one familiar puzzle."""

from .puzzles import PUZZLES, format_grid, parse_grid
from .solver import solve_guided
from .train import train_model


def main() -> None:
    model, report = train_model()
    puzzle = parse_grid(PUZZLES[0].grid)
    result = solve_guided(puzzle, model)
    print(f"blank-cell training accuracy: {report.cell_accuracy:.1%}")
    print("\nPuzzle:\n" + format_grid(puzzle))
    if result.grid is None:
        print("No solution found")
    else:
        print(f"\nSolution ({result.nodes} search nodes):\n" + format_grid(result.grid))


if __name__ == "__main__":
    main()
