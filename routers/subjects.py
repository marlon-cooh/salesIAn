from fastapi import APIRouter, status, HTTPException
from sqlalchemy import select
from models import Subject
from db_create import SessionDep

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