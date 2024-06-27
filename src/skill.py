from __future__ import annotations
from dataclasses import dataclass
import torch


@dataclass
class Skill:
    text: str
    x: float
    y: float
    width: float = 0.0
    height: float = 0.0
    parent: Skill = None

    def rect(self) -> torch.Tensor:
        assert self.width != 0 and self.height != 0
        return torch.tensor([self.x, self.y, self.width, self.height])

    def center(self) -> torch.Tensor:
        assert self.width != 0 and self.height != 0
        return torch.tensor([self.x + self.width / 2, self.y + self.height / 2])
