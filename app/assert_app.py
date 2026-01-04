from fastapi import FastAPI
from app.db_create import create_all_tables
from routers import subjects, students, term

test_app = FastAPI(lifespan=create_all_tables)
test_app.include_router(subjects.router)
test_app.include_router(students.router)
test_app.include_router(term.router)

@test_app.get("/")
async def landing_page():
    return "Hello world!!"