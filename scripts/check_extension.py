"""Check that an in-place extension (run_sweep.py with a larger --t-max) reproduced the
earlier part of each run exactly: the rows with epoch <= the snapshot's t_max in
trajectory.jsonl and endpoint_trajectory.jsonl must equal the snapshot copies taken
before the extension (descent_seconds, a wall-clock time, is ignored). Read-only, fast.

    python scripts/check_extension.py                       # every run in the snapshot dir
    python scripts/check_extension.py --graphs DSJC125.9 miles1500
"""
import argparse, glob, json, os

ap = argparse.ArgumentParser()
ap.add_argument("--main", default="results/main")
ap.add_argument("--snapshot", default="results/main_100k_snapshot")
ap.add_argument("--graphs", nargs="*")
a = ap.parse_args()


def rows(path):
    out = []
    for line in open(path):
        r = json.loads(line)
        r.pop("descent_seconds", None)
        out.append(r)
    return out


n_ok = n_bad = n_pending = 0
for snap in sorted(glob.glob(os.path.join(a.snapshot, "*_seed*"))):
    run = os.path.basename(snap)
    if a.graphs and run.rsplit("_seed", 1)[0] not in a.graphs:
        continue
    old_tmax = json.load(open(os.path.join(snap, "DONE")))["t_max"]
    done = os.path.join(a.main, run, "DONE")
    new_tmax = json.load(open(done))["t_max"] if os.path.exists(done) else None
    if new_tmax is None or new_tmax <= old_tmax:
        print(f"{run}: not extended yet (DONE t_max={new_tmax})"); n_pending += 1
        continue
    diffs = [f for f in ("trajectory.jsonl", "endpoint_trajectory.jsonl")
             if [r for r in rows(os.path.join(a.main, run, f)) if r["epoch"] <= old_tmax]
             != rows(os.path.join(snap, f))]
    new_epochs = sorted({r["epoch"] for r in rows(os.path.join(a.main, run, "endpoint_trajectory.jsonl"))
                         if r["epoch"] > old_tmax})
    if diffs:
        n_bad += 1
        print(f"{run}: DIFFERENT up to epoch {old_tmax} in {diffs}")
    else:
        n_ok += 1
        print(f"{run}: identical up to epoch {old_tmax}; new checkpoints {new_epochs}")
print(f"\nidentical {n_ok}, different {n_bad}, not extended yet {n_pending}")
