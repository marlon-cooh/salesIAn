from fastapi import APIRouter, status, HTTPException
from models import Subject, StudentSubjectLink
from postgres_db_create import SessionDep

router = APIRouter()

@router.post("/subjects", tags=["subjects"])
async def create_subject(subject_info : Subject, session : SessionDep):
    subj_db = Subject.model_validate(subject_info.model_dump())
    session.add(subj_db)
    session.commit()
    session.refresh(subj_db)
    return subj_db

@router.get("/subjects/{subject_id}", tags=["subjects"], response_model=Subject)
async def get_subject(subject_id : int, session : SessionDep):
    subj_db = session.get(Subject, subject_id)
    if not subj_db:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Subject not found")
    return subj_db

@router.post("/subjects/all", tags=["grades"])
async def create_grade_register(subject_info : StudentSubjectLink, session : SessionDep):
    grade_db = StudentSubjectLink.model_validate(subject_info.model_dump())
    session.add(grade_db)
    session.commit()
    session.refresh(grade_db)
    return grade_db

@router.get("/subjects/grades/{subject_id}", tags=["grades"])
async def get_grades_by_subject(subject_id : int, session : SessionDep):
    grades_db = session.get(Subject, subject_id)
    if not grades_db:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Grades not found")
    return grades_db