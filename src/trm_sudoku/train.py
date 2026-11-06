"""Small deterministic training loop."""

import random
from dataclasses import dataclass

import torch
from torch import nn

from .data import make_dataset
from .model import TinyRecursiveModel


@dataclass(frozen=True)
class TrainReport:
    loss: float
    cell_accuracy: float


def train_model(
    epochs: int = 8,
    samples: int = 1024,
    batch_size: int = 64,
    seed: int = 7,
    width: int = 64,
    steps: int = 6,
) -> tuple[TinyRecursiveModel, TrainReport]:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    puzzles, solutions = make_dataset(samples, seed=seed)
    model = TinyRecursiveModel(width=width, steps=steps)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
    criterion = nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(seed)
    loss = torch.tensor(float("nan"))
    for _ in range(epochs):
        for indices in torch.randperm(samples, generator=generator).split(batch_size):
            batch, targets = puzzles[indices], solutions[indices] - 1
            logits = model(batch)
            empty = batch == 0
            loss = criterion(logits[empty], targets[empty])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    with torch.no_grad():
        logits = model(puzzles[:128])
        empty = puzzles[:128] == 0
        accuracy = (logits.argmax(-1)[empty] == (solutions[:128] - 1)[empty]).float().mean()
    return model, TrainReport(float(loss.detach()), float(accuracy))
