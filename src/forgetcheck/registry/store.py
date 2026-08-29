"""Artifact store: checkpoints, activations, and forget-set indices.

Implementation plan §4 and §5. The store is what makes the Kaggle workflow survivable — sessions
are time-limited and disposable, so every run must be independently resumable. A run whose
artifacts already exist is skipped, and a session that dies mid-queue costs only its in-flight
run.

Layout, one subdirectory per artifact kind because Kaggle datasets allow only ~50 top-level
files::

    <root>/
      checkpoints/<run_id>.pt          model weights + metadata
      activations/<run_id>.npz         GAP-pooled probe activations
      forget_sets/<forget_id>.npz      resolved index arrays

**Checkpoints are stored fp32, not fp16.** The implementation plan originally budgeted fp16 to
halve storage. That is the wrong trade here: relearning continues training *from* these
checkpoints, and relearning speed is a core measurement — fp16 weight rounding would perturb the
very trajectory the reversibility audit measures. At ~45 MB each and ~330 checkpoints the full
matrix is ~15 GB against a 200 GB quota, so the saving buys nothing and risks a headline result.
``dtype`` remains configurable for artifacts that will only ever be read by non-training audits.

Every write is atomic (temp file then replace), so a killed session leaves either a complete
artifact or none — never a truncated one that reads as valid and silently corrupts a result.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Iterator, Mapping

import numpy as np

from .ids import parse_run_id
from .provenance import env_sha, file_sha, git_commit, utc_now

__all__ = ["ArtifactStore", "CheckpointMeta", "StoreError"]


class StoreError(RuntimeError):
    """The store was asked for something inconsistent."""


CHECKPOINTS: Final = "checkpoints"
ACTIVATIONS: Final = "activations"
FORGET_SETS: Final = "forget_sets"


@dataclass(frozen=True, slots=True)
class CheckpointMeta:
    """What was saved alongside the weights."""

    run_id: str
    sha: str
    epochs: int | None = None
    train_seed: int | None = None
    final_metrics: Mapping[str, float] | None = None
    hparams_sha: str = ""
    git_commit: str = ""
    env_sha: str = ""
    saved_at: str = ""
    dtype: str = "float32"
    notes: str = ""


class ArtifactStore:
    """Content-addressed-ish store rooted at a directory.

    Not thread-safe, and deliberately not lock-based: concurrency across Kaggle accounts is
    handled by *partitioning the work* (``forgetcheck queue``) rather than by coordinating
    writers. Two sessions computing the same run is a scheduling bug, and the atomic writes mean
    it degrades to wasted compute rather than a corrupt artifact.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)

    # -- paths ---------------------------------------------------------------

    def _dir(self, kind: str) -> Path:
        return self.root / kind

    def checkpoint_path(self, run_id: str) -> Path:
        return self._dir(CHECKPOINTS) / f"{run_id}.pt"

    def meta_path(self, run_id: str) -> Path:
        return self._dir(CHECKPOINTS) / f"{run_id}.json"

    def activations_path(self, run_id: str) -> Path:
        return self._dir(ACTIVATIONS) / f"{run_id}.npz"

    def forget_set_path(self, forget_id: str) -> Path:
        return self._dir(FORGET_SETS) / f"{forget_id}.npz"

    # -- existence -----------------------------------------------------------

    def has_checkpoint(self, run_id: str) -> bool:
        """True only if both weights and metadata are present.

        Both, because a checkpoint without its metadata cannot be provenance-checked, and
        treating it as complete would let an unverifiable artifact into the results.
        """
        return self.checkpoint_path(run_id).is_file() and self.meta_path(run_id).is_file()

    def has_activations(self, run_id: str) -> bool:
        return self.activations_path(run_id).is_file()

    def has_forget_set(self, forget_id: str) -> bool:
        return self.forget_set_path(forget_id).is_file()

    def list_checkpoints(self) -> list[str]:
        d = self._dir(CHECKPOINTS)
        if not d.is_dir():
            return []
        return sorted(p.stem for p in d.glob("*.pt"))

    def iter_checkpoints(self, *, role: str | None = None, forget: str | None = None) -> Iterator[str]:
        """Run ids in the store, optionally filtered by parsed coordinates."""
        for rid in self.list_checkpoints():
            try:
                key = parse_run_id(rid)
            except ValueError:
                continue  # not ours; leave foreign files alone
            if role is not None and key.role != role:
                continue
            if forget is not None and key.forget != forget:
                continue
            yield rid

    # -- checkpoints ---------------------------------------------------------

    def save_checkpoint(
        self,
        run_id: str,
        state_dict: Mapping[str, Any],
        *,
        epochs: int | None = None,
        train_seed: int | None = None,
        final_metrics: Mapping[str, float] | None = None,
        hparams_sha: str = "",
        notes: str = "",
        dtype: str = "float32",
    ) -> CheckpointMeta:
        """Write weights and metadata atomically. Returns the metadata, including the sha."""
        import torch

        parse_run_id(run_id)  # reject a malformed id before writing anything

        if dtype not in {"float32", "float16"}:
            raise StoreError(f"dtype must be float32 or float16, got {dtype!r}")

        target = torch.float32 if dtype == "float32" else torch.float16
        payload = {
            k: (v.detach().to(target).cpu() if hasattr(v, "detach") else v)
            for k, v in state_dict.items()
        }

        ckpt = self.checkpoint_path(run_id)
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        tmp = ckpt.with_suffix(".pt.tmp")
        torch.save(payload, tmp)
        tmp.replace(ckpt)

        meta = CheckpointMeta(
            run_id=run_id,
            sha=file_sha(ckpt),
            epochs=epochs,
            train_seed=train_seed,
            final_metrics=dict(final_metrics) if final_metrics else None,
            hparams_sha=hparams_sha,
            git_commit=git_commit(),
            env_sha=env_sha(),
            saved_at=utc_now(),
            dtype=dtype,
            notes=notes,
        )
        self._write_json(self.meta_path(run_id), _meta_to_dict(meta))
        return meta

    def load_checkpoint(self, run_id: str, *, verify: bool = True) -> tuple[dict, CheckpointMeta]:
        """Load weights and metadata.

        Args:
            verify: re-hash the file and compare against the recorded sha. On by default: an
                audit that silently reads different weights than the ones its record claims is
                exactly the failure this store exists to prevent.
        """
        import torch

        ckpt = self.checkpoint_path(run_id)
        if not ckpt.is_file():
            raise StoreError(f"no checkpoint for {run_id!r} at {ckpt}")

        meta = self.load_meta(run_id)
        if verify:
            actual = file_sha(ckpt)
            if actual != meta.sha:
                raise StoreError(
                    f"{run_id}: checkpoint sha mismatch — file is {actual}, metadata records "
                    f"{meta.sha}. The weights changed after they were recorded; do not audit "
                    "this artifact."
                )

        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        if meta.dtype == "float16":
            state = {k: (v.float() if hasattr(v, "float") else v) for k, v in state.items()}
        return state, meta

    def load_meta(self, run_id: str) -> CheckpointMeta:
        p = self.meta_path(run_id)
        if not p.is_file():
            raise StoreError(f"no checkpoint metadata for {run_id!r} at {p}")
        return _meta_from_dict(json.loads(p.read_text(encoding="utf-8")))

    # -- activations ---------------------------------------------------------

    def save_activations(
        self,
        run_id: str,
        acts: Mapping[str, np.ndarray],
        *,
        probe_ids: np.ndarray | None = None,
        dtype: str = "float16",
    ) -> Path:
        """Store GAP-pooled probe activations, one array per layer.

        fp16 is the right default here (unlike checkpoints): these are read only by similarity
        measures, never trained from, and the storage difference is what keeps the whole matrix
        under 2 GB rather than 4.

        ``probe_ids`` records *which* examples produced the activations. Without it, comparing
        two models' activations assumes they saw the same probes in the same order — an
        assumption that is silently violated the first time someone reshuffles a loader.
        """
        parse_run_id(run_id)
        if not acts:
            raise StoreError(f"{run_id}: refusing to save an empty activation set")

        np_dtype = np.float16 if dtype == "float16" else np.float32
        payload: dict[str, np.ndarray] = {}
        n_rows: set[int] = set()
        for layer, arr in acts.items():
            a = np.asarray(arr)
            if a.ndim != 2:
                raise StoreError(
                    f"{run_id}/{layer}: expected 2-D (n_probe, n_features) after pooling, "
                    f"got shape {a.shape}. Pool over spatial dims before storing — raw conv "
                    "activations are ~130x larger (plan §5)."
                )
            n_rows.add(a.shape[0])
            payload[f"act__{layer}"] = a.astype(np_dtype, copy=False)

        if len(n_rows) != 1:
            raise StoreError(
                f"{run_id}: layers disagree on probe count {sorted(n_rows)}; every layer must "
                "be computed over the same probe set."
            )

        if probe_ids is not None:
            pid = np.asarray(probe_ids)
            if pid.shape[0] != n_rows.pop():
                raise StoreError(f"{run_id}: probe_ids length does not match activation rows")
            payload["probe_ids"] = pid.astype(np.int64, copy=False)

        path = self.activations_path(run_id)
        _atomic_savez(path, payload)
        return path

    def load_activations(self, run_id: str) -> tuple[dict[str, np.ndarray], np.ndarray | None]:
        """Return ``({layer: array}, probe_ids)`` with arrays widened to float32."""
        path = self.activations_path(run_id)
        if not path.is_file():
            raise StoreError(f"no activations for {run_id!r} at {path}")

        with np.load(path) as z:
            acts = {
                k[len("act__"):]: z[k].astype(np.float32)
                for k in z.files
                if k.startswith("act__")
            }
            probe_ids = z["probe_ids"] if "probe_ids" in z.files else None
        return acts, probe_ids

    # -- forget sets ---------------------------------------------------------

    def save_forget_set(
        self, forget_id: str, indices: np.ndarray, *, meta: Mapping[str, Any] | None = None
    ) -> str:
        """Cache a resolved forget-set index array. Returns its sha.

        The spec regenerates these deterministically; the cache exists so that every run can
        *prove* it used the same one rather than assuming it.
        """
        idx = np.asarray(indices, dtype=np.int64)
        if idx.ndim != 1:
            raise StoreError(f"{forget_id}: indices must be 1-D, got shape {idx.shape}")
        if len(np.unique(idx)) != len(idx):
            raise StoreError(f"{forget_id}: indices contain duplicates")
        if not np.all(np.diff(idx) > 0):
            raise StoreError(
                f"{forget_id}: indices must be sorted ascending, so that the array — and "
                "therefore its hash — is canonical."
            )

        path = self.forget_set_path(forget_id)
        _atomic_savez(path, {"indices": idx, "meta": json.dumps(dict(meta or {}))})
        return file_sha(path)

    def load_forget_set(self, forget_id: str) -> tuple[np.ndarray, dict[str, Any]]:
        path = self.forget_set_path(forget_id)
        if not path.is_file():
            raise StoreError(f"no cached forget set {forget_id!r} at {path}")
        with np.load(path) as z:
            return z["indices"].astype(np.int64), json.loads(str(z["meta"]))

    # -- housekeeping --------------------------------------------------------

    def usage(self) -> dict[str, tuple[int, float]]:
        """``{kind: (file_count, megabytes)}`` — for keeping under the Kaggle limits."""
        out: dict[str, tuple[int, float]] = {}
        for kind in (CHECKPOINTS, ACTIVATIONS, FORGET_SETS):
            d = self._dir(kind)
            if not d.is_dir():
                out[kind] = (0, 0.0)
                continue
            files = [p for p in d.iterdir() if p.is_file()]
            total = sum(p.stat().st_size for p in files)
            out[kind] = (len(files), round(total / 1e6, 1))
        return out

    def export(self, dest: str | Path, *, kinds: tuple[str, ...] = (CHECKPOINTS,)) -> Path:
        """Copy selected artifact kinds to ``dest`` for upload as a Kaggle dataset version."""
        target = Path(dest)
        target.mkdir(parents=True, exist_ok=True)
        for kind in kinds:
            src = self._dir(kind)
            if src.is_dir():
                shutil.copytree(src, target / kind, dirs_exist_ok=True)
        return target

    @staticmethod
    def _write_json(path: Path, obj: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(path)

    def __repr__(self) -> str:
        return f"ArtifactStore({self.root})"


def _atomic_savez(path: Path, payload: Mapping[str, Any]) -> None:
    """Write a compressed .npz atomically.

    ``np.savez_compressed`` appends ``.npz`` to any path that does not already end in it, which
    silently renames a temp file out from under a subsequent ``replace``. Passing an open handle
    makes numpy write exactly where it is told.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with open(tmp, "wb") as fh:
            np.savez_compressed(fh, **payload)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def _meta_to_dict(m: CheckpointMeta) -> dict[str, Any]:
    return {
        "run_id": m.run_id,
        "sha": m.sha,
        "epochs": m.epochs,
        "train_seed": m.train_seed,
        "final_metrics": dict(m.final_metrics) if m.final_metrics else None,
        "hparams_sha": m.hparams_sha,
        "git_commit": m.git_commit,
        "env_sha": m.env_sha,
        "saved_at": m.saved_at,
        "dtype": m.dtype,
        "notes": m.notes,
    }


def _meta_from_dict(d: Mapping[str, Any]) -> CheckpointMeta:
    return CheckpointMeta(
        run_id=d["run_id"],
        sha=d["sha"],
        epochs=d.get("epochs"),
        train_seed=d.get("train_seed"),
        final_metrics=d.get("final_metrics"),
        hparams_sha=d.get("hparams_sha", ""),
        git_commit=d.get("git_commit", ""),
        env_sha=d.get("env_sha", ""),
        saved_at=d.get("saved_at", ""),
        dtype=d.get("dtype", "float32"),
        notes=d.get("notes", ""),
    )
