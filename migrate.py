from auscult.db import engine
from auscult.models import Base


def main() -> None:
    Base.metadata.create_all(engine)
    print("Tables created (or already existed).")


if __name__ == "__main__":
    main()
