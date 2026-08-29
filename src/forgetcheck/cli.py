"""``forgetcheck`` — the single entry point every notebook calls.

Kaggle notebooks are thin on purpose: install the package at a pinned commit, run one command
here, push the artefacts. The implementations live in version-controlled, tested modules rather
than in notebook cells, which is what makes ``git_commit`` in every record mean something and
what stops twelve accounts from each carrying their own drifting copy of the training loop.

The command that makes several accounts usable at once is :func:`cmd_queue`. It enumerates the
work a stage requires, partitions it deterministically, and runs only this account's share —
with no coordination between accounts, because run identifiers are pure functions of the config
and every account therefore derives the identical work list.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

from .config import Context, find_configs
from .data.forget_sets import spec_by_id
from .registry import run_id
from .train import base_task, oracle_task, run_task, shadow_task
from .unlearn import CORE_METHODS, base_run_id_for, method_names, run_unlearn

__all__ = ["main", "build_parser", "plan_stage", "shard", "WorkItem"]

FULL = "full"


# --------------------------------------------------------------------------- work items


@dataclass(frozen=True, slots=True)
class WorkItem:
    """One unit of work: an identity, a description, and a thunk that performs it."""

    run_id: str
    kind: str
    run: Callable[[], object]

    def sort_key(self) -> str:
        """Deterministic ordering, independent of how the plan happened to be built."""
        return self.run_id


def _base_items(ctx: Context) -> list[WorkItem]:
    items = []
    variants = [FULL] + [f for f in ctx.all_forget_ids() if f.startswith("canary")]
    for variant in variants:
        for seed in ctx.seeds["train"]:
            task = base_task(ctx.bundle, seed=seed, variant=variant)
            items.append(
                WorkItem(
                    task.run_id,
                    "base",
                    lambda t=task: run_task(
                        t, ctx.train_config, store=ctx.store,
                        records_dir=ctx.records_dir, device=ctx.device,
                        eval_bundle=ctx.bundle,
                    ),
                )
            )
    return items


def _oracle_items(ctx: Context) -> list[WorkItem]:
    items = []
    for forget_id in ctx.all_forget_ids():
        spec = ctx.spec(forget_id)
        fidx = ctx.forget_indices(forget_id)

        # Paired oracles: one per M0 seed, sharing that M0's initialisation so the pair differs
        # only in the data it saw.
        for seed in ctx.base["oracles"]["paired_seeds"]:
            task = oracle_task(ctx.bundle, spec, fidx, seed=seed)
            items.append(
                WorkItem(
                    task.run_id, "oracle-paired",
                    lambda t=task, f=fidx: run_task(
                        t, ctx.train_config, store=ctx.store,
                        records_dir=ctx.records_dir, device=ctx.device,
                        eval_bundle=ctx.bundle, forget_indices=f,
                    ),
                )
            )

        # Ensemble oracles exist only at the primary condition: they estimate the reference
        # distribution and the oracle-vs-oracle baselines, which the calibration is anchored on.
        if forget_id == ctx.primary_condition:
            for oseed in ctx.seeds["oracle"]:
                task = oracle_task(ctx.bundle, spec, fidx, seed=oseed, ensemble=True)
                items.append(
                    WorkItem(
                        task.run_id, "oracle-ensemble",
                        lambda t=task, f=fidx: run_task(
                            t, ctx.train_config, store=ctx.store,
                            records_dir=ctx.records_dir, device=ctx.device,
                            eval_bundle=ctx.bundle, forget_indices=f,
                        ),
                    )
                )
    return items


def _shadow_items(ctx: Context) -> list[WorkItem]:
    n = ctx.base["shadows"]["count"]
    frac = ctx.base["shadows"]["subset_fraction"]
    audit_seed = ctx.seeds["audit"]
    items = []
    for i in range(n):
        task = shadow_task(ctx.bundle, i, audit_seed=audit_seed, fraction=frac)
        items.append(
            WorkItem(
                task.run_id, "shadow",
                lambda t=task: run_task(
                    t, ctx.train_config, store=ctx.store,
                    records_dir=ctx.records_dir, device=ctx.device,
                    eval_bundle=ctx.bundle,
                ),
            )
        )
    return items


def _unlearn_items(ctx: Context, methods: Sequence[str] | None = None) -> list[WorkItem]:
    methods = tuple(methods or CORE_METHODS)
    items = []
    for forget_id in ctx.all_forget_ids():
        spec = ctx.spec(forget_id)
        fidx = ctx.forget_indices(forget_id)
        for method in methods:
            for seed in ctx.seeds["train"]:
                rid = run_id(role="unlearn", forget=forget_id, method=method, seed=seed)
                items.append(
                    WorkItem(
                        rid, "unlearn",
                        lambda s=spec, f=fidx, m=method, sd=seed: run_unlearn(
                            method=m, spec=s, forget_indices=f, bundle=ctx.bundle,
                            seed=sd, store=ctx.store, records_dir=ctx.records_dir,
                            device=ctx.device,
                            batch_size=ctx.train_config.batch_size,
                        ),
                    )
                )
    return items


STAGES: dict[int, tuple[str, Callable[[Context], list[WorkItem]]]] = {
    3: ("base models and oracles", lambda c: _base_items(c) + _oracle_items(c)),
    4: ("RMIA shadow models", _shadow_items),
    5: ("unlearning runs", _unlearn_items),
}


def shard(items: Sequence[WorkItem], *, account: int, of: int) -> list[WorkItem]:
    """This account's share of the work: every ``of``-th item of a sorted list.

    Striping a canonically sorted list rather than hashing each id. Hashing is stable when new
    work is added later, but it distributes badly at small counts — 32 shadow models across 12
    accounts left one account with nothing to do and another with four. Striping is exactly
    balanced, and its instability is harmless here: :func:`_execute` skips anything already in
    the store, so a redistributed item is skipped rather than recomputed.

    Every account derives the identical list, because run identifiers are pure functions of the
    configuration. No coordination is needed, and no two accounts are given the same item.
    """
    if of < 1 or not (1 <= account <= of):
        raise ValueError(f"account must be in [1, {of}], got {account}")
    ordered = sorted(items, key=WorkItem.sort_key)
    return ordered[account - 1 :: of]


def plan_stage(ctx: Context, stage: int) -> list[WorkItem]:
    try:
        _label, builder = STAGES[stage]
    except KeyError:
        raise SystemExit(
            f"unknown stage {stage}; queueable stages are {sorted(STAGES)}"
        ) from None
    return builder(ctx)


# --------------------------------------------------------------------------- commands


def _ctx(args) -> Context:
    configs = Path(args.configs) if args.configs else find_configs(args.root)
    return Context(configs=configs, device=args.device, root=Path(args.root))


def _execute(items: Iterable[WorkItem], *, dry_run: bool, store) -> int:
    items = list(items)
    done = skipped = failed = 0
    for i, item in enumerate(items, 1):
        exists = store.has_checkpoint(item.run_id)
        if dry_run:
            mark = "have" if exists else "todo"
            print(f"  [{mark}] {item.run_id}")
            skipped += exists
            continue
        if exists:
            skipped += 1
            continue
        print(f"[{i}/{len(items)}] {item.run_id} ...", flush=True)
        t0 = time.perf_counter()
        try:
            item.run()
        except Exception as exc:  # keep the queue moving; one bad cell is not the whole session
            failed += 1
            print(f"    FAILED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            continue
        done += 1
        print(f"    done in {time.perf_counter() - t0:.1f}s", flush=True)

    verb = "would run" if dry_run else "ran"
    print(f"\n{verb} {done}, skipped {skipped} already present, {failed} failed")
    return 1 if failed else 0


def cmd_queue(args) -> int:
    """Run this account's share of a stage."""
    ctx = _ctx(args)
    items = shard(plan_stage(ctx, args.stage), account=args.account, of=args.of)
    label = STAGES[args.stage][0]
    print(
        f"stage {args.stage} ({label}): {len(items)} items "
        f"for account {args.account} of {args.of}\n"
    )
    return _execute(items, dry_run=args.dry_run, store=ctx.store)


def cmd_train_base(args) -> int:
    ctx = _ctx(args)
    task = base_task(ctx.bundle, seed=args.seed, variant=args.variant)
    return _execute(
        [WorkItem(task.run_id, "base", lambda: run_task(
            task, ctx.train_config, store=ctx.store, records_dir=ctx.records_dir,
            device=ctx.device, eval_bundle=ctx.bundle))],
        dry_run=args.dry_run, store=ctx.store,
    )


def cmd_train_oracle(args) -> int:
    ctx = _ctx(args)
    spec = ctx.spec(args.forget)
    fidx = ctx.forget_indices(args.forget)
    task = oracle_task(ctx.bundle, spec, fidx, seed=args.seed, ensemble=args.ensemble)
    return _execute(
        [WorkItem(task.run_id, "oracle", lambda: run_task(
            task, ctx.train_config, store=ctx.store, records_dir=ctx.records_dir,
            device=ctx.device, eval_bundle=ctx.bundle, forget_indices=fidx))],
        dry_run=args.dry_run, store=ctx.store,
    )


def cmd_train_shadows(args) -> int:
    ctx = _ctx(args)
    items = _shadow_items(ctx)[args.start : args.start + args.count]
    return _execute(items, dry_run=args.dry_run, store=ctx.store)


def cmd_unlearn(args) -> int:
    ctx = _ctx(args)
    spec = ctx.spec(args.forget)
    fidx = ctx.forget_indices(args.forget)
    rid = run_id(role="unlearn", forget=args.forget, method=args.method, seed=args.seed)
    return _execute(
        [WorkItem(rid, "unlearn", lambda: run_unlearn(
            method=args.method, spec=spec, forget_indices=fidx, bundle=ctx.bundle,
            seed=args.seed, store=ctx.store, records_dir=ctx.records_dir,
            device=ctx.device, batch_size=ctx.train_config.batch_size))],
        dry_run=args.dry_run, store=ctx.store,
    )


def cmd_status(args) -> int:
    """What is in the store, and what each stage still needs."""
    ctx = _ctx(args)
    for k, v in ctx.describe().items():
        print(f"  {k:20s} {v}")

    usage = ctx.store.usage()
    print("\nartifact store:")
    for kind, (n, mb) in usage.items():
        print(f"  {kind:14s} {n:5d} files  {mb:9.1f} MB")

    print("\nstage progress:")
    for stage, (label, _b) in STAGES.items():
        try:
            items = plan_stage(ctx, stage)
        except Exception as exc:
            print(f"  stage {stage} ({label}): unavailable — {exc}")
            continue
        have = sum(1 for it in items if ctx.store.has_checkpoint(it.run_id))
        bar = "#" * int(20 * have / max(1, len(items)))
        print(f"  stage {stage} {label:26s} {have:4d}/{len(items):4d} |{bar:<20}|")
    return 0


def cmd_forget_sets(args) -> int:
    """Materialise and describe every forget condition."""
    ctx = _ctx(args)
    mem = ctx.memorization
    print(f"{'condition':18s} {'kind':11s} {'n':>6s}  {'mean mem':>9s}")
    for forget_id in ctx.all_forget_ids():
        spec = ctx.spec(forget_id)
        idx = ctx.forget_indices(forget_id)
        print(f"{forget_id:18s} {spec.kind:11s} {len(idx):6d}  {mem[idx].mean():9.4f}")
    return 0


# --------------------------------------------------------------------------- parser


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="forgetcheck",
        description="Audit disagreement, reversibility and audit validity in machine unlearning.",
    )
    p.add_argument("--configs", default=None, help="path to the configs/ directory")
    p.add_argument("--root", default=".", help="project root for data and artifact paths")
    p.add_argument("--device", default="cpu", help="cpu or cuda")
    p.add_argument("--dry-run", action="store_true", help="list the work without running it")
    sub = p.add_subparsers(dest="command", required=True)

    q = sub.add_parser("queue", help="run this account's share of a stage")
    q.add_argument("--stage", type=int, required=True, choices=sorted(STAGES))
    q.add_argument("--account", type=int, default=1, help="1-based account index")
    q.add_argument("--of", type=int, default=1, help="how many accounts share this stage")
    q.set_defaults(func=cmd_queue)

    b = sub.add_parser("train-base", help="train one original model")
    b.add_argument("--seed", type=int, required=True)
    b.add_argument("--variant", default=FULL, help="'full' or a canary condition id")
    b.set_defaults(func=cmd_train_base)

    o = sub.add_parser("train-oracle", help="train one retrained oracle")
    o.add_argument("--forget", required=True)
    o.add_argument("--seed", type=int, required=True)
    o.add_argument("--ensemble", action="store_true",
                   help="an ensemble member (oracle seed) rather than a paired oracle")
    o.set_defaults(func=cmd_train_oracle)

    s = sub.add_parser("train-shadows", help="train RMIA reference models")
    s.add_argument("--start", type=int, default=0)
    s.add_argument("--count", type=int, default=8)
    s.set_defaults(func=cmd_train_shadows)

    u = sub.add_parser("unlearn", help="apply one unlearning method")
    u.add_argument("--forget", required=True)
    u.add_argument("--method", required=True, choices=sorted(method_names()))
    u.add_argument("--seed", type=int, required=True)
    u.set_defaults(func=cmd_unlearn)

    st = sub.add_parser("status", help="what is in the store and what each stage still needs")
    st.set_defaults(func=cmd_status)

    fs = sub.add_parser("forget-sets", help="materialise and describe every forget condition")
    fs.set_defaults(func=cmd_forget_sets)

    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "queue" and not (1 <= args.account <= args.of):
        raise SystemExit(f"--account must be in [1, {args.of}], got {args.account}")
    return args.func(args) or 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
