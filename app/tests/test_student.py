from fastapi import status

def test_create_student(client):
    response = client.post(
        "/students/",
        json={
            "name": "Juliana Bonita",
            "age": 20,
            "social_data": 1,
            "overall_grade": "SUPERIOR",
            "fundamental_comp": "SUPERIOR",
            "behavioral_comp": "SUPERIOR",
            "disabilities": "null"
        }
    )
    assert response.status_code == status.HTTP_201_CREATED
    
def test_read_student(client):
    response = client.post(
        "/students/",
        json={
            "name": "Juliana Bonita",
            "age": 20,
            "social_data": 1,
            "overall_grade": "SUPERIOR",
            "fundamental_comp": "SUPERIOR",
            "behavioral_comp": "SUPERIOR",
            "disabilities": "null"
        }
    )

    student_id : int = response.json()["id"]
    response_read = client.get(f"/students/{student_id}")
    assert response_read.status_code == status.HTTP_200_OK
    assert response_read.json()["name"] == "Juliana Bonita"
    
def test_delete_student(client):
    pass
    