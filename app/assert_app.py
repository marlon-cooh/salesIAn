from fastapi import FastAPI
from app.dependencies import lifespan
from routers import subjects, students, term

test_app = FastAPI(lifespan=lifespan)
test_app.include_router(subjects.router)
test_app.include_router(students.router)
test_app.include_router(term.router)

@test_app.get("/")
async def landing_page():
    return "Hello world!!"