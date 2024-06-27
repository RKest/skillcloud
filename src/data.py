from __future__ import annotations
import pandas as pd
import torch

from src.scraping import load_skills, fill_skills
from src.draw import measure, Skill

import os
from math import isnan
from dataclasses import dataclass
from typing import List, Tuple
from random import uniform

PKL_FILE = "skills.pkl"


@dataclass
class SkillsTensor:
    def __init__(self, skills: List[Skill]):
        self.x = torch.tensor([[float(s.x), float(s.y)] for s in skills])
        self.x_dim = torch.tensor([[float(s.width), float(s.height)] for s in skills])
        self.links = self.init_links(skills)

    @staticmethod
    def init_links(skills: List[Skill]) -> List[Tuple[int, int]]:
        result: List[Tuple[int, int]] = []
        parents = {}

        for i, s in enumerate(skills):
            if s.parent is None:
                parents[s.text] = i

        for i, s in enumerate(skills):
            if s.parent is not None:
                result.append((parents[s.parent.text], i))

        return result


def get_skills_df():
    if os.path.exists(PKL_FILE):
        return pd.read_pickle(PKL_FILE)

    skills = load_skills("skills.yaml")
    skills: pd.DataFrame = skills.apply(lambda x: x.explode(ignore_index=True)).map(fill_skills)
    skills.to_pickle("skills.pkl")
    return skills


def unpack_skills(df: pd.DataFrame) -> List[Skill]:
    result: List[Skill] = []
    parents = {}
    cols = df.columns.tolist()
    for c in cols:
        new_skill = Skill(text=c, x=0, y=0)
        result.append(new_skill)
        parents[c] = new_skill

    for _, row in df.iterrows():
        for c in cols:
            cell = row[c]
            if isinstance(cell, float) and isnan(cell):
                continue
            new_skill = Skill(text=cell["name"], x=0, y=0, parent=parents[c])
            result.append(new_skill)

    return result


def get_skills() -> SkillsTensor:
    skills_df = get_skills_df()
    unpacked = unpack_skills(skills_df)
    measured = measure(unpacked)
    return SkillsTensor(measured)


def visualize(xs: List[Tuple[float, float]]) -> None:
    ss = get_skills_df()
    ss = unpack_skills(ss)
    for s, x in zip(ss, xs):
        s.x, s.y = x[0], x[1]
    _ = measure(ss)
    while True:
        pass
