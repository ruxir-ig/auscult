"""Run the end-to-end integration test suite."""

import pytest


def main() -> None:
    raise SystemExit(pytest.main(["-v", "tests/test_integration.py"]))


if __name__ == "__main__":
    main()
