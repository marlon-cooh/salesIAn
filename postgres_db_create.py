from typing import Annotated
from fastapi import Depends, FastAPI
import psycopg2
import os
from dotenv import load_dotenv
from urllib import parse
from sqlmodel import SQLModel, create_engine, Session
from models import *
from contextlib import asynccontextmanager

# Loading env variables
load_dotenv()
X = ["DB_USER", "DB_PASSWORD", "DB_HOST", "DB_PORT", "DB_NAME"]

DB_USER = os.getenv(X[0])
DB_PASSWORD = os.getenv(X[1])
DB_HOST = os.getenv(X[2])
DB_PORT = os.getenv(X[3])
DB_NAME = os.getenv(X[4]).strip().lower()

safe_password = parse.quote_plus(DB_PASSWORD)

DATABASE_URL = f"postgresql://{DB_USER}:{safe_password}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

def test_connection_psql():
    
    # Testing every env value exists.
    missing_env = {
        var : value for var, value in zip(X, [DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME]) if not value
    }
    if missing_env:
        raise ValueError ("Invalid Database Connection keywords.")
    
    try:
        conn = psycopg2.connect(DATABASE_URL)
        print(f"✅ Successfully connected to 'laholanda' using psycopg2")
        
        cursor = conn.cursor()
        cursor.execute("SELECT current_database();")
        print(f"PostgreSQL version : {cursor.fetchone()}")
        
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"❌ Error: {e}")
        
        
engine = create_engine(DATABASE_URL, echo=True)

@asynccontextmanager
async def create_db_and_tables(app: FastAPI):
    # This creates the tables in the DB if they don't exist
    print("🚀 Initializing database and creating tables...")
    SQLModel.metadata.create_all(engine)
    yield
    print("🛑 Shutting down and cleaning up resources...")

def get_session():
    with Session(engine) as session:
        yield session

SessionDep = Annotated[Session, Depends(get_session)]

if __name__ == "__main__":
    test_connection_psql()
    
    