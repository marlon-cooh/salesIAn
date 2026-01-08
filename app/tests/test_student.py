from fastapi import status
import json
import os
from sqlmodel import select
from models import Student, Subject, Term, StudentSubjectLink

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
student_path = os.path.join(
    os.path.dirname(os.path.dirname(BASE_DIR)),
    "student_info.json"
)
subject_path = os.path.join(
    os.path.dirname(os.path.dirname(BASE_DIR)),
    "subjects.json"
)
student_to_subject_path = os.path.join(
    os.path.dirname(os.path.dirname(BASE_DIR)),
    "student_to_subject.json"
)
term_path = os.path.join(
    os.path.dirname(os.path.dirname(BASE_DIR)),
    "terms.json"
)

# Students info.
with open(student_path, "r", encoding="utf-8") as f:
    student_info = json.load(f)
    
# Subjects info.
with open(subject_path, "r", encoding="utf-8") as f:
    subject_info = json.load(f)
    
# Student to subject info.
with open(student_to_subject_path, "r", encoding="utf-8") as f:
    student_to_subject_info = json.load(f)

# Terms
with open(term_path, "r", encoding="utf-8") as f:
    term_info = json.load(f)
    
# Students tests.  
def test_create_student(client):
    response = client.post(
        "/students/",
        json={
            "code" : 10,
            "name": "Juliana Bonita",
            "age": 20,
            "social_data": 1,
            "overall_grade": "HIGH",
            "fundamental_comp": 10.0,
            "behavioral_comp": 10.0,
            "disabilities": "null"
        }
    )
    assert response.status_code == status.HTTP_201_CREATED
    
def test_read_student(client):
    response = client.post(
        "/students/",
        json={
            "code" : 10,
            "name": "Juliana Bonita",
            "age": 20,
            "social_data": 1,
            "overall_grade": "HIGH",
            "fundamental_comp": 10.0,
            "behavioral_comp": 10.0,
            "disabilities": "null"
        }
    )

    student_id : int = response.json()["id"]
    response_read = client.get(f"/students/{student_id}")
    assert response_read.status_code == status.HTTP_200_OK
    assert response_read.json()["name"] == "Juliana Bonita"
    
def test_create_list_students(client):
    response = client.post("/students/cohort", json=student_info)
    assert response.status_code == status.HTTP_201_CREATED
    assert len(response.json()) == len(student_info)
    
# Subjects.
def test_create_subject(client):
    response = client.post(
        "/subjects/",
        json={
            "component": "Mathematics"
        }
        )
    assert response.status_code == status.HTTP_200_OK
    
def test_create_list_subjects(client):
    subjects = subject_info
    for subj in subjects:
        response = client.post("/subjects/", json=subj)
        assert response.status_code == status.HTTP_200_OK
    
def test_read_all_subjects(client):
    response = client.get("/subjects/")
    assert response.status_code == status.HTTP_200_OK
    
def test_create_subject_link(session):
    student = Student(
        code=101,
        name="Juliana Bonilla",
        age=20,
        social_data=1,
        overall_grade="HIGH",
        fundamental_comp=9,
        behavioral_comp=10,
        disabilities=None,
    )
    subject = Subject(component="QUIMICA")
    term = Term(code="P1", label="Periodo 1", order=1)
    session.add(student)
    session.add(subject)
    session.add(term)
    session.commit()
    session.refresh(student)
    session.refresh(subject)
    session.refresh(term)

    link = StudentSubjectLink(
        student_id=student.id,
        subject_id=subject.id,
        term_id=term.id,
        grade="SUPERIOR",
    )
    session.add(link)
    session.commit()

    stored_link = session.exec(
        select(StudentSubjectLink).where(
            StudentSubjectLink.student_id == student.id,
            StudentSubjectLink.subject_id == subject.id,
            StudentSubjectLink.term_id == term.id,
        )
    ).one()

    assert stored_link.grade == "SUPERIOR"

# def test_delete_student(client):
#     pass

# Student to subject
def test_create_student_to_subject(client):
    response = client.post(
        "/subjects/relationship",
        json={
        "student_id": 28,
        "subject_id": 10,
        "term_id": 1,
        "grade": 3.0
        },
    )
    assert response.status_code == status.HTTP_201_CREATED
    assert response.json()["student_id"] == 28

# def test_create_list_students_to_subject(client):
#     response = client.post(
#         "/subjects/all",
#         json=student_to_subject_info
#     )
#     assert response.status_code == status.HTTP_201_CREATED
    
# Term
def test_create_term(client):
    response = client.post(
        "/term",
        json={
        "code": "P1",
        "label": "1\u00b0 Periodo",
        "order": 1
        }
    )
    assert response.status_code == status.HTTP_201_CREATED
    
def test_create_list_terms(client):
    response = client.post(
        "/term/all",
        json=term_info
    )
    assert response.status_code == status.HTTP_201_CREATED