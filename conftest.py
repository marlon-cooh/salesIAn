import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel

from app.main import test_app
from db_create import get_session

# Database for testing
sqlite_name = "db.sqlite3"
sqlite_url = f"sqlite:///{sqlite_name}"

engine = create_engine(
    sqlite_url,
    connect_args={"check_same_thread" : False},
    poolclass=StaticPool
)

@pytest.fixture(name="session")
def session_fixture():
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        yield session
    SQLModel.metadata.drop_all(engine)
    
@pytest.fixture(name="client")
def client_fixture(session: Session):
    def get_session_override():
        return session

    test_app.dependency_overrides[get_session] = get_session_override
    client = TestClient(test_app)
    yield client
    test_app.dependency_overrides.clear()
    
