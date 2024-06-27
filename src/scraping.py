import pandas as pd
import requests
import yaml
from os import getenv
from typing import TypedDict
from typing_extensions import NotRequired

GITHUB_TOKEN = getenv("GITHUB_TOKEN")
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"


class PartialSkill(TypedDict):
    name: NotRequired[str]
    link: str
    score: NotRequired[int]


class Skill(TypedDict):
    name: str
    link: str
    score: int


def load_skills(path: str) -> pd.DataFrame:
    with open(path, "r") as f:
        return pd.json_normalize(yaml.safe_load(f))


def github_stars(url: str) -> int:
    if "github.com" not in url:
        return -1
    api_url = url.replace("github.com", "api.github.com/repos")
    res = requests.get(api_url, headers={"User-Agent": USER_AGENT, "Authorization": F"Bearer {GITHUB_TOKEN}"})
    assert res.ok, F"{res.text=} {url=}"
    return int(res.json()["stargazers_count"])


def github_name(url: str) -> str:
    if "github.com" not in url:
        return "unknown"
    return url.rsplit('/', 1)[-1]


def skill_from_github(url: str) -> Skill:
    return Skill(name=github_name(url), link=url, score=github_stars(url))


def skill_from_partial(skill: PartialSkill) -> Skill:
    link = skill["link"]
    score = skill["score"] if "score" in skill else github_stars(link)
    name = skill["name"] if "name" in skill else github_name(link)
    return Skill(name=name, link=link, score=score)


def skill_from_url(skill_url: str) -> Skill:
    if "github" in skill_url:
        return skill_from_github(skill_url)
    raise Exception(F"Don't know how to create a skill solely based on url: {skill_url}")


def fill_skills(skill: PartialSkill | float | str) -> dict | float:
    match skill:
        case str():
            return skill_from_url(skill)
        case dict():
            return skill_from_partial(skill)
        case _:
            print(F"{type(skill)=}")
            return float("nan")

