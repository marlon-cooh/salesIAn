from fastapi import APIRouter, status, HTTPException
from models import Subject, StudentSubjectLink, SubjectUpdate
from app.dependencies import SessionDep
from sqlmodel import select

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

@router.get("/subjects", tags=["subjects"], response_model=list[Subject])
async def get_all_subjects(session : SessionDep):
    subjects = session.exec(select(Subject)).all()
    return subjects

@router.put("/subjects/{subject_id}", tags=["subjects"], response_model=Subject)
def update_subject(subject_id : int, subject_update : SubjectUpdate, session : SessionDep):
    subject = session.get(Subject, subject_id)
    if not subject:
        raise HTTPException(status_code=404, detail="Subject not found.")
    subject.sqlmodel_update(subject_update.model_dump(exclude_unset=True))
    session.add(subject)
    session.commit()
    session.refresh(subject)
    return subject

@router.delete("/subjects/{subject_id}", tags=["subjects"])
def delete_subject(subject_id : int, session : SessionDep):
    subject = session.get(Subject, subject_id)
    if not subject:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Subject not registered."
        )
    session.delete(subject)
    session.commit()
    return {"ok" : True}

@router.post("/subjects/all", tags=["subjects"], status_code=status.HTTP_201_CREATED)
async def create_multiple_subjects(subject_info: list[Subject], session : SessionDep):
    session.add_all(subject_info)
    session.commit()
    return subject_info

# StudentSubjectLink table    
@router.post("/subjects/relationship", tags=["grades"], response_model=StudentSubjectLink, status_code=status.HTTP_201_CREATED)
async def create_grade_register(subject_info : StudentSubjectLink, session : SessionDep):
    grade_db = StudentSubjectLink.model_validate(subject_info.model_dump())
    session.add(grade_db)
    session.commit()
    session.refresh(grade_db)
    return grade_db

@router.post("/subjects/links", tags=["grades"], status_code=status.HTTP_201_CREATED)
async def create_grade_register(subject_info: list[StudentSubjectLink], session: SessionDep):
    session.add_all(subject_info)
    session.commit()
    return subject_info

@router.get("/subjects/grades/{subject_id}", tags=["grades"])
async def get_grades_by_subject(subject_id : int, session : SessionDep):
    grades_db = session.get(Subject, subject_id)
    if not grades_db:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Grades not found")
    return grades_db