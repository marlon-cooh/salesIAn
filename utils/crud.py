#!/usr/bin/env python3

import os
import json, requests

fastapi_route = "http://127.0.0.1:8000"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load(name):
    with open(os.path.join(BASE_DIR, name), "r", encoding="utf-8") as f:
        return json.load(f)

terms = load("terms.json")
subjects = load("subjects.json")
students = load("student_info.json")
links = load("student_to_subject.json")

def post(path, payload):
    r = requests.post(f"{fastapi_route}{path}", json=payload, timeout=120)
    print(path, r.status_code)
    if r.status_code >= 400:
        print(r.text)
        raise SystemExit(1)
    return r.json()

if __name__ == "__main__":
    # 1) Terms
    post("/term/all", terms)

    # 2) Subjects
    post("/subjects/all", subjects)

    # 3) Students
    post("/students/cohort", students)

    # 4) Links (grades)
    post("/subjects/links", links)