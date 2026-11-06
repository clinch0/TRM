"""A tiny weight-tied recursive refinement network for 9x9 Sudoku."""

import torch
from torch import nn


class RefinementBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.update = nn.Sequential(
            nn.Linear(width * 4, width * 2),
            nn.ReLU(),
            nn.Linear(width * 2, width),
        )
        self.norm = nn.LayerNorm(width)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        batch, _, width = state.shape
        board = state.reshape(batch, 9, 9, width)
        row = board.mean(dim=2, keepdim=True).expand_as(board)
        column = board.mean(dim=1, keepdim=True).expand_as(board)
        boxes = board.reshape(batch, 3, 3, 3, 3, width)
        boxes = boxes.mean(dim=(2, 4), keepdim=True).expand_as(boxes).reshape_as(board)
        context = torch.cat((board, row, column, boxes), dim=-1).reshape(batch, 81, width * 4)
        return self.norm(state + self.update(context))


class TinyRecursiveModel(nn.Module):
    """Reuse one small block several times, then predict a digit per cell."""

    def __init__(self, width: int = 64, steps: int = 8):
        super().__init__()
        self.steps = steps
        self.value_embedding = nn.Embedding(10, width)
        self.position_embedding = nn.Parameter(torch.randn(81, width) * 0.02)
        self.block = RefinementBlock(width)
        self.head = nn.Linear(width, 9)

    def forward(self, grid: torch.Tensor, steps: int | None = None) -> torch.Tensor:
        if grid.ndim == 1:
            grid = grid.unsqueeze(0)
        if grid.shape[1:] != (81,):
            raise ValueError("expected grids with shape [batch, 81]")
        state = self.value_embedding(grid) + self.position_embedding
        for _ in range(self.steps if steps is None else steps):
            state = self.block(state)
        return self.head(state)

    @torch.no_grad()
    def predict(self, grid: torch.Tensor) -> torch.Tensor:
        prediction = self(grid).argmax(dim=-1) + 1
        original = grid.unsqueeze(0) if grid.ndim == 1 else grid
        return torch.where(original > 0, original, prediction)
