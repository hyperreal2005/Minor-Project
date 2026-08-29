"""Shared fixtures and graceful skipping.

Most of the suite runs on synthetic data and needs nothing on disk. A few tests need the real
CIFAR-10 bundle, which is 170 MB and deliberately not committed. Rather than fail for a teammate
who has not downloaded it yet — or on a CI runner where the download is flaky — those tests skip
with a message that says exactly what to do.

The memorization scores *are* committed (400 KB), so nothing needs to skip on their account.
"""

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CIFAR_DIR = REPO / "data" / "cifar-10-batches-py"
MEMORIZATION = REPO / "data" / "memorization" / "cifar10_memorization.npy"


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "requires_data: needs the real CIFAR-10 bundle on disk"
    )


def pytest_collection_modifyitems(config, items):
    """Skip data-dependent tests when CIFAR-10 has not been downloaded."""
    if CIFAR_DIR.is_dir():
        return
    skip = pytest.mark.skip(
        reason=(
            "CIFAR-10 not downloaded. Run:  python -c "
            "\"from forgetcheck.data.cifar import load_cifar; load_cifar('data')\""
        )
    )
    for item in items:
        if "requires_data" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO


@pytest.fixture(scope="session")
def has_real_data() -> bool:
    return CIFAR_DIR.is_dir() and MEMORIZATION.is_file()
