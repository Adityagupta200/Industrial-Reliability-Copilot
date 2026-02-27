from __future__ import annotations

from rag_service.db.session import engine
from rag_service.db.models import Base


def init_db() -> None:
    Base.metadata.create_all(bind=engine)


if __name__ == "__main__":
    init_db()
    print("DB initialized.")
