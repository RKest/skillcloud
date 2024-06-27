from __future__ import annotations

from selenium import webdriver
from selenium.webdriver.support.wait import WebElement
from selenium.webdriver.common.by import By
from jinja2 import Environment, FileSystemLoader, select_autoescape
from typing import List

from src.skill import Skill

jinja_env = Environment(
    loader=FileSystemLoader("templates/"),
    autoescape=select_autoescape()
)

driver = None


def fill_measurements(skill: Skill, el: WebElement, parent: WebElement) -> Skill:
    assert skill.width == 0.0 and skill.height == 0.0
    skill.width = el.size["width"] / parent.size["width"]
    skill.height = el.size["height"] / parent.size["height"]
    return skill


def measure(skills: List[Skill]) -> List[Skill]:
    global driver
    options = webdriver.ChromeOptions()
    options.add_argument("headless")
    driver = driver or webdriver.Chrome()
    driver.maximize_window()
    template = jinja_env.get_template("template.html")
    html = template.render(skills=skills)
    driver.get(F"data:text/html;charset=utf-8,{html}")
    skills_parent = driver.find_element(By.TAG_NAME, "main")
    skill_elements = driver.find_elements(By.TAG_NAME, "div")
    driver.save_screenshot("ss.png")
    return [
        fill_measurements(skill, el, skills_parent)
        for skill, el in zip(skills, skill_elements)
    ]
