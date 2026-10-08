"""Step D (manifest) + Step H ordering: write configs/sweep_manifest.csv, the primary
DIMACS test set (proven chi, chi >= 3) in the order the main sweep runs it: smallest
first by (n, m, name). Row index = Slurm array task id in scripts/run_sweep.sbatch.
Re-running only rewrites the file, so changing the benchmark is an explicit, diffable act.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    df = pd.read_csv(ROOT / "data" / "dimacs" / "chi_bounds.csv")
    df = df[df["chi_proven"] == True]  # noqa: E712
    df = df[df["chi"] >= 3].copy()
    df["chi"] = df["chi"].astype(int)
    df["k"] = df["chi"] - 1
    df["density"] = 2 * df["m"] / (df["n"] * (df["n"] - 1))
    df = df.sort_values(["n", "m", "name"]).reset_index(drop=True)
    df.index.name = "idx"
    cols = ["name", "family", "n", "m", "chi", "k", "density", "max_degree", "n_isolated"]
    out = ROOT / "configs" / "sweep_manifest.csv"
    df[cols].to_csv(out, float_format="%.6f")
    print(f"wrote {out} ({len(df)} graphs, n from {df.n.iloc[0]} to {df.n.iloc[-1]})")


if __name__ == "__main__":
    main()
