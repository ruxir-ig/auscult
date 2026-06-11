from auscult.db import get_engine
from auscult.models import Base


def main() -> None:
    Base.metadata.create_all(get_engine())
    print("Tables created (or already existed).")


if __name__ == "__main__":
    main()
