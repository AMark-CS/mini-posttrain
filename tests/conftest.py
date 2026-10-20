"""Pytest config — keep tests runnable without any GPU or model downloads."""
import sys
from pathlib import Path

import pytest

# Add the project root to sys.path so ``import nano_posttrain`` works
# when pytest is run from anywhere.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def pytest_addoption(parser):
    parser.addoption(
        "--run-smoke",
        action="store_true",
        default=False,
        help="Run the optional smoke tests that need to download a tiny model.",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-smoke"):
        return
    skip_smoke = pytest.mark.skip(reason="needs --run-smoke to run")
    for item in items:
        if "smoke" in item.keywords:
            item.add_marker(skip_smoke)
