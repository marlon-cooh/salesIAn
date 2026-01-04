from fastapi.testclient import TestClient

def test_student(student):
    assert type(student) == TestClient