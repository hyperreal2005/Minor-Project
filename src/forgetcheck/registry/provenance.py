"""Provenance capture: which code, which environment, which weights.

Every record carries enough provenance to answer "what produced this number?" months later,
from a Kaggle session that no longer exists. Three things are captured:

``git_commit``
    The exact code revision. Kaggle notebooks install the package at a pinned commit, so this
    is the link between a result and the source that produced it.

``env_sha``
    A hash over the versions of the packages that can change a number. Catches the case where
    two members get different results from identical code because one has a different torch.

``checkpoint_sha``
    Content hash of the weights an audit actually read. This is what makes the "same forget set,
    same model" claim checkable rather than assumed.

Nothing here raises. Provenance capture must never be the reason a 40-minute training run fails
to record its result — a missing git repository degrades to the ``nogit`` sentinel and is
recorded honestly rather than crashing or silently pretending.
"""

from __future__ import annotations

import functools
import hashlib
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

__all__ = [
    "git_commit",
    "env_sha",
    "env_details",
    "file_sha",
    "bytes_sha",
    "config_sha",
    "utc_now",
    "NO_GIT",
    "SHA_LEN",
]

#: Sentinel recorded when the working tree is not a git repository. Expected before the repo is
#: pushed; if it appears in results produced after that, provenance is broken and the run should
#: be treated as unreproducible.
NO_GIT: Final = "nogit"

#: Truncation length for all recorded hashes. 16 hex chars = 64 bits, ample for collision-free
#: identification within a project of this size, and short enough to read in a table.
SHA_LEN: Final = 16

#: Packages whose version can change a result. Anything not here is presumed not to matter.
_TRACKED_PACKAGES: Final = (
    "torch",
    "torchvision",
    "numpy",
    "scipy",
    "scikit-learn",
    "pyarrow",
)


def utc_now() -> str:
    """Current time as an ISO-8601 UTC string, second resolution."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


# --------------------------------------------------------------------------- git


@functools.lru_cache(maxsize=4)
def git_commit(repo: str | Path | None = None, *, allow_dirty_marker: bool = True) -> str:
    """Return the current commit sha, or :data:`NO_GIT`.

    A dirty working tree is reported as ``<sha>-dirty`` so that results produced from
    uncommitted code are visibly distinguishable in the record table.
    """
    root = Path(repo) if repo is not None else Path.cwd()

    def _run(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", *args],
                cwd=root,
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if out.returncode != 0:
            return None
        return out.stdout.strip()

    sha = _run("rev-parse", "HEAD")
    if not sha:
        return NO_GIT

    short = sha[:SHA_LEN]
    if allow_dirty_marker:
        status = _run("status", "--porcelain")
        if status:  # non-empty means uncommitted changes
            return f"{short}-dirty"
    return short


# --------------------------------------------------------------------------- environment


def _package_version(name: str) -> str:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return "absent"


@functools.lru_cache(maxsize=1)
def env_details() -> dict[str, str]:
    """The environment facts that go into :func:`env_sha`, for human inspection.

    Deliberately reads package versions from installation metadata rather than importing the
    packages. Importing torch to interrogate it costs ~9 seconds and would be paid by every
    process that writes a record, including pure-analysis sessions that never touch a model.

    The accelerator is *not* included here. It is a property of where the code ran, not of what
    software the environment contains, and it belongs in :func:`runtime_context` — keeping it
    out is also what makes ``env_sha`` identical for the same environment whether or not torch
    happens to be loaded.
    """
    details = {
        "python": platform.python_version(),
        "platform": f"{platform.system()}-{platform.machine()}",
    }
    for pkg in _TRACKED_PACKAGES:
        details[pkg] = _package_version(pkg)
    return details


@functools.lru_cache(maxsize=1)
def env_sha() -> str:
    """Stable hash over :func:`env_details`."""
    d = env_details()
    payload = "\n".join(f"{k}={d[k]}" for k in sorted(d))
    return hashlib.sha256(payload.encode()).hexdigest()[:SHA_LEN]


# --------------------------------------------------------------------------- content hashing


def bytes_sha(data: bytes) -> str:
    """Truncated sha256 of a byte string."""
    return hashlib.sha256(data).hexdigest()[:SHA_LEN]


def file_sha(path: str | Path, *, chunk: int = 1 << 20) -> str:
    """Truncated sha256 of a file's contents, streamed so large checkpoints stay cheap."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()[:SHA_LEN]


def config_sha(obj: object) -> str:
    """Stable hash of a config object.

    Dicts are hashed by sorted key so that two members whose YAML loaders order keys differently
    still agree. This is what ``hparams_sha`` records: the resolved hyperparameters a run
    actually used, not the file it nominally read.
    """
    return bytes_sha(_canonical(obj).encode())


def _canonical(obj: object) -> str:
    if isinstance(obj, dict):
        inner = ",".join(f"{k!r}:{_canonical(obj[k])}" for k in sorted(obj, key=repr))
        return "{" + inner + "}"
    if isinstance(obj, (list, tuple)):
        return "[" + ",".join(_canonical(v) for v in obj) + "]"
    if isinstance(obj, set):
        return "{" + ",".join(sorted(_canonical(v) for v in obj)) + "}"
    if isinstance(obj, float):
        # repr() round-trips exactly and keeps 0.1 from hashing differently across platforms.
        return repr(obj)
    return repr(obj)


# --------------------------------------------------------------------------- context


def runtime_context() -> dict[str, str]:
    """Where this process is running, and on what accelerator.

    Separate from :func:`env_sha` on purpose (see :func:`env_details`). Call this from training
    and audit entry points — where torch is loaded anyway — and record it in ``notes`` when the
    device could matter. CPU and CUDA can differ in the last digits, and for this project the
    CPU/GPU split across machines is a real workflow rather than a hypothetical.
    """
    on_kaggle = bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE")) or Path("/kaggle").exists()
    on_colab = "google.colab" in sys.modules
    ctx = {
        "host": "kaggle" if on_kaggle else ("colab" if on_colab else "local"),
        "node": platform.node(),
        "device": "unknown",
    }

    # Only interrogate torch if it is already imported: this function must stay cheap enough to
    # call freely, and importing torch here would reintroduce the cost env_details avoids.
    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            ctx["device"] = (
                f"cuda:{torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "cpu"
            )
        except Exception:
            ctx["device"] = "unknown"
    return ctx
