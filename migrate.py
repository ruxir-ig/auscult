from alembic import command
from alembic.config import Config


def main() -> None:
    command.upgrade(Config("alembic.ini"), "head")
    print("Migrations applied.")


if __name__ == "__main__":
    main()
