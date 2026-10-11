"""Diagnostic for the TabuCol run (docs/progress_exp1.md §8 rows 17, 19): on the register-
allocation graphs TabuCol (random start, k = chi - 1) found its best within ~100-1,000 iterations
and then never improved in 3-4 million, ending far above DSATUR (e.g. inithx.i.1: 30 vs ~1).
This separates "implementation bug" from "search weakness" with fixed-budget cases:

  bench   DSJC125.5 at k = 17 (= chi): published TabuCol finds a proper 17-colouring
  chi     fpsol2.i.1 / inithx.i.1 / mulsol.i.1 at k = chi (a proper colouring exists; DSATUR finds it)
  dsatur  the same graphs at k = chi - 1 started from deterministic DSATUR's colouring

Each case runs in its own process for --seconds. Read-only with respect to results/.

    python scripts/diag_tabucol.py --seconds 60
"""
import argparse, json, os, subprocess, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from graphsml.controls import dsatur_fixed_k  # noqa: E402
from graphsml.dimacs import load_col_graph  # noqa: E402
from graphsml.endpoint import EndpointGraph, conflict_count  # noqa: E402
from graphsml.tabucol import tabucol  # noqa: E402

CASES = [("bench", "DSJC125.5", 0), ("chi", "fpsol2.i.1", 0), ("chi", "inithx.i.1", 0), ("chi", "mulsol.i.1", 0),
         ("dsatur", "fpsol2.i.1", -1), ("dsatur", "inithx.i.1", -1), ("dsatur", "mulsol.i.1", -1)]


def run_case(i: int, seconds: float):
    kind, name, dk = CASES[i]
    man = pd.read_csv(ROOT / "configs" / "sweep_manifest.csv")
    chi = int(man[man.name == name].chi.iloc[0])
    k = chi + dk
    eg = EndpointGraph.from_networkx(load_col_graph(ROOT / "data" / "dimacs" / "col" / f"{name}.col"))
    start = None
    if kind == "dsatur":
        start = dsatur_fixed_k(eg, k)
    res = tabucol(eg, k, seed=0, start=start, time_budget=seconds)
    assert res.best_conflicts == conflict_count(eg, res.best_colours)
    print(json.dumps(dict(case=kind, graph=name, k=k, chi=chi, start=res.start_conflicts, best=res.best_conflicts,
                          iterations=res.iterations, iter_at_best=res.iter_at_best,
                          seconds_at_best=round(res.seconds_at_best, 1), stopped_by=res.stopped_by,
                          history_tail=res.history[-5:])), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--case", type=int)
    a = ap.parse_args()
    if a.case is not None:
        return run_case(a.case, a.seconds)
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    procs = [subprocess.Popen([sys.executable, "-u", __file__, "--case", str(i), "--seconds", str(a.seconds)], env=env)
             for i in range(len(CASES))]
    print("exit codes", [p.wait() for p in procs], flush=True)


if __name__ == "__main__":
    main()
