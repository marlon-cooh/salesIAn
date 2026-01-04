# app/dependencies.py
import os
from contextlib import asynccontextmanager
from typing import Annotated
from fastapi import Depends, FastAPI
from sqlmodel import Session, create_engine, SQLModel
# from app.postgres_db_create import DATABASE_URL
from dotenv import load_dotenv
from urllib.parse import quote_plus

def build_database_url() -> str:
    """
    Switch DB backend with APP_DB=sqlite or APP_DB=postgres.
        - sqlite uses a local file db.sqlite3
        - postgres uses DATABASE_URL
    """
    # Default production database.
    backend = os.getenv("APP_DB", "postgres").lower()

    # Default testing database.
    if backend == "sqlite":
        sqlite_name = os.getenv("SQLITE_NAME", "db.sqlite3")
        return f"sqlite:///{sqlite_name}"
    
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        user = os.getenv("DB_USER")
        password = os.getenv("DB_PASSWORD")
        host = os.getenv("DB_HOST", "127.0.0.1")
        port = os.getenv("DB_PORT", "5432")
        name = os.getenv("DB_NAME")
        if not all([user, password, name]):
            raise ValueError("Missing DATABASE_URL or DB_USER/DB_PASSWORD/DB_NAME")
        
        password = quote_plus(password)
        db_url = f"postgresql://{user}:{password}@{host}:{port}/{name}"
        
    return db_url

if __name__ == "__main__":
    load_dotenv()
    DATABASE_URL = build_database_url()
    print("DATABASE_URL:", DATABASE_URL)
    