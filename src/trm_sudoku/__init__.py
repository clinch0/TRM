"""A small recursive neural model and an honest Sudoku evaluation harness."""

from .model import TinyRecursiveModel
from .solver import is_valid_solution, solve_guided

__all__ = ["TinyRecursiveModel", "is_valid_solution", "solve_guided"]
