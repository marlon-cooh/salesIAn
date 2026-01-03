from fastapi import FastAPI
from postgres_db_create import create_db_and_tables
from db_create import create_all_tables
from routers import subjects, students, term

# test_app
test_app = FastAPI(lifespan=create_all_tables)

app = FastAPI(lifespan=create_db_and_tables)
app.include_router(subjects.router)
app.include_router(students.router)
app.include_router(term.router)

@app.get("/")
async def landing_page():
    return "Hello world!!"