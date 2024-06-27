import torch
import matplotlib.pyplot as plt
from tqdm import trange

from random import uniform
from dataclasses import dataclass
from typing import List, Tuple


def out_of_bounds(output: torch.Tensor, dims: torch.Tensor) -> torch.Tensor:
    acc = torch.tensor(0.0)
    o, d = output.view([-1, 2]), dims.view([-1, 2])
    for out, dim in zip(o, d):
        x1, x2 = out[0], out[0] + dim[0]
        y1, y2 = out[1], out[1] + dim[1]
        acc += torch.max(x1 - 1., torch.tensor(0.0)) + torch.max(-x1, torch.tensor(0.0))
        acc += torch.max(x2 - 1., torch.tensor(0.0)) + torch.max(-x2, torch.tensor(0.0))
        acc += torch.max(y1 - 1., torch.tensor(0.0)) + torch.max(-y1, torch.tensor(0.0))
        acc += torch.max(y2 - 1., torch.tensor(0.0)) + torch.max(-y2, torch.tensor(0.0))
    return acc


def iou2(box1, box2):
    box1_x1, box1_y1 = box1[0], box1[1]
    box1_x2, box1_y2 = box1[0] + box1[2], box1[1] + box1[3]

    box2_x1, box2_y1 = box2[0], box2[1]
    box2_x2, box2_y2 = box2[0] + box2[2], box2[1] + box2[3]

    x_left = torch.max(box1_x1, box2_x1)
    y_top = torch.max(box1_y1, box2_y1)
    x_right = torch.min(box1_x2, box2_x2)
    y_bottom = torch.min(box1_y2, box2_y2)

    intersection = torch.clamp(x_right - x_left, min=0) * torch.clamp(y_bottom - y_top, min=0)

    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
    union = box1_area + box2_area - intersection
    iou = intersection / union

    return iou


def iou(rects: torch.Tensor) -> torch.Tensor:
    return iou2(rects[0], rects[1])


def combinations_2d(tensor):
    n = tensor.shape[0]
    idx = torch.combinations(torch.arange(n), r=2)
    result = tensor[idx]
    return result


def overlap(output: torch.Tensor, dims: torch.Tensor) -> torch.Tensor:
    assert output.shape == dims.shape
    o_padded = output - torch.mean(output).div(10.).expand_as(output)
    d_padded = dims + torch.mean(dims).div(10.).expand_as(dims)
    o, d = o_padded.view([-1, 2]), d_padded.view([-1, 2])
    oc, od = combinations_2d(o), combinations_2d(d)
    rects = torch.cat([oc, od], dim=2)
    viou = torch.func.vmap(iou)
    return torch.sum(viou(rects))


def center(xy: torch.Tensor, dims: torch.Tensor) -> torch.Tensor:
    return (xy + dims / 2).unsqueeze(0)


def distance(output: torch.Tensor, dims: torch.Tensor, links: List[Tuple[int, int]]):
    total_dist = torch.tensor(0.0)
    o, d = output.view([-1, 2]), dims.view([-1, 2])
    for i, j in links:
        dist = torch.cdist(center(o[i], d[i]), center(o[j], d[j]))
        total_dist += dist[0, 0]
    return total_dist


def minimize(output: torch.Tensor, dims: torch.Tensor) -> torch.Tensor:
    xy2 = output + dims
    x1 = output.view([-1, 2])[:, 0]
    y1 = output.view([-1, 2])[:, 1]
    x2 = xy2.view([-1, 2])[:, 0]
    y2 = xy2.view([-1, 2])[:, 1]
    xs = torch.cat([x1, x2])
    ys = torch.cat([y1, y2])
    return torch.max(xs) - torch.min(xs) + torch.max(ys) - torch.min(ys)


def loss(output: torch.Tensor, dims: torch.Tensor, links: List[Tuple[int, int]]) -> torch.Tensor:
    return out_of_bounds(output, dims) + distance(output, dims, links) + overlap(output, dims) + minimize(output, dims)


def draw(output: torch.Tensor, dims: torch.Tensor):
    plt.figure(figsize=(8, 8))
    rects = []
    for s, d in zip(output, dims):
        print(F"{s=}, {d=}")
        rects.append(
            plt.Rectangle(xy=(s[0], s[1]), width=d[0], height=d[1], color=tuple(uniform(0, 1) for _ in range(3))))
    fig = plt.gcf()
    ax = fig.gca()
    for r in rects:
        ax.add_patch(r)
    plt.xlim([-1, 2])
    plt.ylim([-1, 2])
    plt.show()


class Net(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.l1 = torch.nn.Linear(dim, dim * 2)
        self.a1 = torch.nn.Sigmoid()
        self.l2 = torch.nn.Linear(dim * 2, dim)
        self.a2 = torch.nn.Sigmoid()

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        y = self.l1.forward(y)
        y = self.a1.forward(y)
        y = self.l2.forward(y)
        y = self.a2.forward(y)
        return y


@dataclass
class Gym:
    x: torch.Tensor
    x_dim: torch.Tensor
    links: List[Tuple[int, int]]

    def __post_init__(self):
        self.size = len(self.x.flatten())
        self.model = Net(self.size)
        self.opt = torch.optim.SGD(self.model.parameters(), lr=0.0001)

    def train(self):
        for _ in trange(10000):
            self.train_step()

    def train_step(self):
        self.opt.zero_grad()
        y = self.model.forward(self.x)
        self.loss = loss(y, self.x_dim, self.links)
        self.loss.backward()
        self.opt.step()
        self.x = y.detach()

        return self.x.view([-1, 2]).tolist()

    def print_loss(self):
        print(F"{self.loss=}")

    def print_summary(self):
        print("-----------------------------------")
        print(F"{self.y=}")
        print(F"{overlap(self.x, self.x_dims)=}")
        print(F"{out_of_bounds(self.x, self.x_dims)=}")
        print(F"{distance(self.x, self.x_dims, self.x_links)=}")
        print("-----------------------------------")
