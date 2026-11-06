import torch

from trm_sudoku.data import make_dataset
from trm_sudoku.model import TinyRecursiveModel
from trm_sudoku.puzzles import PUZZLES, parse_grid
from trm_sudoku.solver import is_valid_solution, solve_guided


def test_model_shape_and_weight_tying():
    model = TinyRecursiveModel(width=16, steps=2)
    puzzle = parse_grid(PUZZLES[0].grid)
    assert model(puzzle).shape == (1, 81, 9)
    assert sum(1 for name, _ in model.named_modules() if name.endswith("block")) == 1


def test_generated_solutions_are_valid():
    puzzles, solutions = make_dataset(12)
    assert puzzles.shape == solutions.shape == (12, 81)
    assert all(is_valid_solution(solution) for solution in solutions)


def test_exact_solver_handles_fixed_real_puzzles():
    for item in PUZZLES:
        puzzle = parse_grid(item.grid)
        result = solve_guided(puzzle)
        assert result.grid is not None
        assert is_valid_solution(result.grid, puzzle)
        assert torch.equal(result.grid, parse_grid(item.solution))


def test_validator_rejects_changed_clue():
    puzzle = parse_grid(PUZZLES[0].grid)
    solution = parse_grid(PUZZLES[0].solution)
    solution[0] = 6
    assert not is_valid_solution(solution, puzzle)
