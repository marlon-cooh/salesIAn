from fastapi import FastAPI
from db_create import create_all_tables
from routers import subjects, students

app = FastAPI(lifespan=create_all_tables)
app.include_router(subjects.router)
app.include_router(students.router)

@app.get("/")
async def landing_page():
    return "Hello world!!"