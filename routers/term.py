from fastapi import APIRouter, status, HTTPException
from models import Term
from app.dependencies import SessionDep

router = APIRouter()

@router.post("/term", tags=["terms"], status_code=status.HTTP_201_CREATED)
async def create_term(term_info : Term, session : SessionDep):
    term_db = Term.model_validate(term_info.model_dump())
    session.add(term_db)
    session.commit()
    session.refresh(term_db)
    return term_db

@router.post("/term/all", tags=["terms"], status_code=status.HTTP_201_CREATED)
async def create_terms(term_info: list[Term], session: SessionDep):
    session.add_all(term_info)
    session.commit()
    return term_info

@router.get("/term/{term_id}", tags=["terms"], response_model=Term)
async def get_term(term_id : int, session : SessionDep):
    term_db = session.get(Term, term_id)
    if not term_db:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Subject not found")
    return term_db