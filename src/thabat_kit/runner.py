"""Run the same seeds under several numerical environments and keep every row.

run_fn(seed, env) must return a dict with at least `outcome` (0 or 1). The keys
`score` and `param_hash` are used by the analysis when present. The runner
applies the environment, seeds everything, times the call and appends one JSON
line per (env, seed). Existing lines are skipped, so an interrupted run resumes.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Iterable

from .env import NumericalEnv, capture_env
from .seeding import seed_everything


class PairedTable:
    """Rows indexed by environment name and seed."""

    def __init__(self, rows: list[dict]):
        self._rows: dict[str, dict[int, dict]] = {}
        for row in rows:
            self._rows.setdefault(row["env"], {})[int(row["seed"])] = row

    def envs(self) -> list[str]:
        return list(self._rows)

    def rows(self, env: str) -> dict[int, dict]:
        return self._rows.get(env, {})

    def seeds(self, env: str) -> list[int]:
        return sorted(self.rows(env))

    def flat(self):
        for per_env in self._rows.values():
            yield from per_env.values()

    @classmethod
    def load(cls, *paths) -> "PairedTable":
        rows: list[dict] = []
        for p in paths:
            p = Path(p)
            if p.exists():
                rows += [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
        return cls(rows)


class PairedRunner:
    def __init__(self, run_fn: Callable[[int, NumericalEnv], dict], envs: Iterable[NumericalEnv], path):
        self.run_fn = run_fn
        self.envs = list(envs)
        names = [e.name for e in self.envs]
        if len(set(names)) != len(names):
            raise ValueError("environment names must be unique")
        self.path = Path(path)
        self.env_path = self.path.with_suffix(".envs.json")

    def run(self, seeds: Iterable[int], log: Callable[[str], None] | None = None) -> PairedTable:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        done = {(r["env"], int(r["seed"])) for r in PairedTable.load(self.path).flat()} if self.path.exists() else set()
        captures = json.loads(self.env_path.read_text()) if self.env_path.exists() else {}
        for seed in seeds:
            for env in self.envs:
                if (env.name, int(seed)) in done:
                    continue
                env.apply()
                bundle = seed_everything(seed)
                if env.name not in captures:
                    captures[env.name] = capture_env(env)
                    self.env_path.write_text(json.dumps(captures, indent=2, sort_keys=True))
                t0 = time.perf_counter()
                result = self.run_fn(int(seed), env)
                if "outcome" not in result:
                    raise KeyError("run_fn must return a dict with an 'outcome' key")
                row = {"seed": int(seed), "env": env.name, **result, "seconds": round(time.perf_counter() - t0, 3)}
                with self.path.open("a") as fh:
                    fh.write(json.dumps(row, sort_keys=True) + "\n")
                if log:
                    log(f"seed={seed} env={env.name} outcome={row['outcome']} seconds={row['seconds']}")
        return PairedTable.load(self.path)

