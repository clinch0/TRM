"""Fixed 9x9 Sudoku examples used only for evaluation."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Puzzle:
    name: str
    grid: str
    solution: str


PUZZLES = (
    Puzzle(
        "classic",
        "530070000600195000098000060800060003400803001700020006060000280000419005000080079",
        "534678912672195348198342567859761423426853791713924856961537284287419635345286179",
    ),
    Puzzle(
        "wikipedia",
        "000260701680070090190004500820100040004602900050003028009300074040050036703018000",
        "435269781682571493197834562826195347374682915951743628519326874248957136763418259",
    ),
    Puzzle(
        "ai-escargot",
        "100007090030020008009600500005300900010080002600004000300000010040000007007000300",
        "162857493534129678789643521475312986913586742628794135356478219241935867897261354",
    ),
)


def parse_grid(text: str) -> torch.Tensor:
    if len(text) != 81 or any(char not in "0123456789." for char in text):
        raise ValueError("a Sudoku grid must contain exactly 81 digits or dots")
    return torch.tensor([0 if char == "." else int(char) for char in text], dtype=torch.long)


def format_grid(grid: torch.Tensor) -> str:
    rows = []
    values = grid.reshape(9, 9).tolist()
    for row, cells in enumerate(values):
        if row in {3, 6}:
            rows.append("------+-------+------")
        chunks = [" ".join(str(value or ".") for value in cells[i : i + 3]) for i in (0, 3, 6)]
        rows.append(" | ".join(chunks))
    return "\n".join(rows)
