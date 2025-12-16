from fastapi import FastAPI, Depends
from typing import Annotated
from sqlmodel import Session, create_engine, SQLModel

# Instancing sqlite database.
sqlite_name = "db.sqlite3"
sqlite_url = f"sqlite:///{sqlite_name}"

engine = create_engine(sqlite_url)

async def create_all_tables(app: FastAPI):
    SQLModel.metadata.create_all(engine)
    yield
    
def get_session():
    with Session(engine) as session:
        yield session

SessionDep = Annotated[Session, Depends(get_session)]