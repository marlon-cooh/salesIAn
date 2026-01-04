from models import Student, StudentCreate, StudentUpdate, StudentSubjectLink
from sqlmodel import select
from postgres_db_create import SessionDep
from fastapi import APIRouter, HTTPException, status

router = APIRouter()

# Post student info
@router.post("/students/", response_model=Student, status_code=status.HTTP_201_CREATED, tags=['student'])
def create_student(student: StudentCreate, session: SessionDep):
    db_student = Student.model_validate(student.model_dump())
    session.add(db_student)
    session.commit()
    session.refresh(db_student)
    return db_student

# Get student info
@router.get("/students/{student_id}", response_model=Student, tags=['student'], status_code=status.HTTP_200_OK)
def read_student(student_id: int, session: SessionDep):
    student = session.get(Student, student_id)
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")
    return student

# Modify student info.
@router.put("/students/{student_id}", response_model=Student, tags=['student'])
def update_student(student_id: int, student_update: StudentUpdate, session: SessionDep):
    student = session.get(Student, student_id)
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")
    student.sqlmodel_update(student_update.model_dump(exclude_unset=True))
    session.add(student)
    session.commit()
    session.refresh(student)
    return student

# Delete student info.
@router.delete("/students/{student_id}", tags=['student'])
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

@router.get("/students/", response_model=list[Student], tags=['student'])
def read_students(session: SessionDep):
    students = session.exec(select(Student)).all()
    return students 