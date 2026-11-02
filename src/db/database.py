import logging
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

logger = logging.getLogger("edgevision.database")

# Environment variable with PostgreSQL default and SQLite local fallback
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://edge_user:edge_password@localhost:5432/edgevision_db"
)

# If PostgreSQL is not reachable in local developer mode, use local SQLite database
try:
    if DATABASE_URL.startswith("sqlite"):
        engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
    else:
        engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_size=10, max_overflow=20)
except Exception as exc:
    logger.warning("Failed to configure database engine for '%s': %s. Falling back to SQLite.", DATABASE_URL, exc)
    engine = create_engine("sqlite:///./edgevision.db", connect_args={"check_same_thread": False})

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    """Dependency generator that provides a database session per request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Initializes schema tables in the database."""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database schema initialized successfully.")
    except Exception as exc:
        logger.error("Database schema initialization encountered error: %s", exc)
