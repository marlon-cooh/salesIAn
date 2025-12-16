from models import Student, StudentCreate, StudentUpdate
from sqlmodel import select
from db_create import SessionDep
from fastapi import APIRouter, HTTPException, status

router = APIRouter()

@router.post("/students/", response_model=Student)
def create_student(student: StudentCreate, session: SessionDep):
    db_student = Student.model_validate(student.model_dump())
    session.add(db_student)
    session.commit()
    session.refresh(db_student)
    return db_student

@router.get("/students/{student_id}", response_model=Student)
def read_student(student_id: int, session: SessionDep):
    student = session.get(Student, student_id)
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")
    return student

# @router.put("/students/{student_id}", response_model=Student)
# def update_student(student_id: int, student_update: StudentUpdate, session: SessionDep):
#     student = session.get(Student, student_id)
#     if not student:
#         raise HTTPException(status_code=404, detail="Student not found")
#     student_data = student_update.dict(exclude_unset=True)
#     for key, value in student_data.items():
#         setattr(student, key, value)
#     session.add(student)
#     session.commit()
#     session.refresh(student)
#     return student

@router.delete("/students/{student_id}")
def delete_student(student_id: int, session: SessionDep):
    student = session.get(Student, student_id)
    if not student:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail="Student not found"
        )
    session.delete(student)
    session.commit()
    return {"ok": True}

@router.get("/students/", response_model=list[Student])
def read_students(session: SessionDep):
    students = session.exec(select(Student)).all()
    return students 