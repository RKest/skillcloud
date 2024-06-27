from flask import Flask, send_from_directory, render_template, request
from src.data import get_skills, visualize, get_skills_df, unpack_skills, SkillsTensor
from src.ai import Gym
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from random import uniform

app = Flask(__name__)
exec = ThreadPoolExecutor(1)
current_skills = None
gym: Gym = None
running = False
mutex = Lock()


@app.route("/static/<path:path>")
def root(path):
    return send_from_directory("static", path=path)


@app.route("/init")
def init():
    assert request.method == "GET"
    global current_skills
    skills_df = get_skills_df()
    current_skills = unpack_skills(skills_df)
    for s in current_skills:
        s.x, s.y = uniform(0.0, 0.7), uniform(0.0, 0.9)
    return render_template("template.html", skills=current_skills)


@app.route("/measure", methods=["POST"])
def measure():
    assert request.method == "POST"
    global current_skills, gym
    json = request.get_json()
    for s in current_skills:
        s.width, s.height = json[s.text]["x"], json[s.text]["y"]
    skills_tensor = SkillsTensor(current_skills)
    gym = Gym(skills_tensor.x.flatten(), skills_tensor.x_dim.flatten(), skills_tensor.links)
    return "ok"


@app.route("/data")
def inc():
    global current_skills, exec, running, mutex
    if not running:
        running = True
        exec.submit(run_training)
    mutex.acquire()
    rendered = render_template("partial.html", skills=current_skills)
    mutex.release()
    return rendered


def run_training():
    global current_skills, gym, mutex
    for i in range(10000):
        new_xs = gym.train_step()
        mutex.acquire()
        for ss, xs in zip(current_skills, new_xs):
            ss.x, ss.y = xs[0], xs[1]
        mutex.release()
