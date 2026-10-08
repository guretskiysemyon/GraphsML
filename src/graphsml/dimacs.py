"""Loader for DIMACS .col graph files.

Format (see data/dimacs/README.binformat):
    c <comment line>
    p edge <n> <m>
    e <u> <v>       (1-indexed, each undirected edge may appear once or twice)
"""

from __future__ import annotations

from pathlib import Path

import networkx as nx


def load_col_graph(path: str | Path) -> nx.Graph:
    """Parse a DIMACS .col file into a 0-indexed, simple, undirected networkx Graph.

    Duplicate and self-loop edges in the file are dropped silently: the DIMACS
    p-line edge count is sometimes twice the unique edge count (each edge listed
    both directions), and networkx's Graph de-duplicates on insertion anyway.
    """
    path = Path(path)
    g = nx.Graph()
    declared_n = None
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("c"):
                continue
            tokens = line.split()
            if tokens[0] == "p":
                # p edge <n> <m>
                declared_n = int(tokens[2])
                g.add_nodes_from(range(declared_n))
            elif tokens[0] == "e":
                u, v = int(tokens[1]) - 1, int(tokens[2]) - 1
                if u != v:
                    g.add_edge(u, v)
    if declared_n is not None and g.number_of_nodes() != declared_n:
        raise ValueError(
            f"{path}: p-line declares {declared_n} nodes but graph has "
            f"{g.number_of_nodes()} (isolated nodes should still have been added)"
        )
    return g
