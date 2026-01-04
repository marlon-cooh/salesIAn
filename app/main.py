from fastapi import FastAPI
from postgres_db_create import create_db_and_tables
from routers import subjects, students, term

app = FastAPI(lifespan=create_db_and_tables)
app.include_router(subjects.router)
app.include_router(students.router)
app.include_router(term.router)

@app.get("/")
async def landing_page():
    return "Hello world!!"