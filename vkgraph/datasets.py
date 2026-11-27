"""vkgraph.datasets: Built-in classic teaching graphs (Zachary's Karate Club)."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Union
import urllib.request
import numpy as np
import vknumpy as vnp

from .graph import Graph


def karate_club() -> Tuple[Graph, np.ndarray]:
    """Return Zachary's famous 34-node, 78-edge Karate Club social network and faction labels (0 = Mr. Hi, 1 = Officer)."""
    edges = [
        (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 10), (0, 11),
        (0, 12), (0, 13), (0, 17), (0, 19), (0, 21), (0, 31),
        (1, 2), (1, 3), (1, 7), (1, 13), (1, 17), (1, 19), (1, 21), (1, 30),
        (2, 3), (2, 7), (2, 8), (2, 9), (2, 13), (2, 27), (2, 28), (2, 32),
        (3, 7), (3, 12), (3, 13),
        (4, 6), (4, 10), (5, 6), (5, 10), (5, 16), (6, 16),
        (8, 30), (8, 32), (8, 33), (9, 33), (13, 33),
        (14, 32), (14, 33), (15, 32), (15, 33), (18, 32), (18, 33),
        (19, 33), (20, 32), (20, 33), (22, 32), (22, 33),
        (23, 25), (23, 27), (23, 29), (23, 32), (23, 33),
        (24, 25), (24, 27), (24, 31), (25, 31), (26, 29), (26, 33),
        (27, 33), (28, 31), (28, 33), (29, 32), (29, 33), (30, 32), (30, 33),
        (31, 32), (31, 33), (32, 33),
    ]
    # Ground-truth faction after the real 1977 club split (0 = Mr. Hi's faction, 1 = Officer's faction)
    factions = np.array(
        [
            0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0,
            0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        ],
        dtype=np.int32,
    )
    g = Graph(
        num_nodes=34,
        edges=edges,
        directed=False,
        name="Zachary's Karate Club",
        node_metadata=[{"member": i, "faction": "Mr. Hi" if f == 0 else "Officer"} for i, f in enumerate(factions)],
    )
    return g, factions


def load_cora(data_dir: Union[str, Path] = "data/cora") -> Tuple[Graph, vnp.ndarray, List[str]]:
    """Download (once) and load the real Cora AI Citation Network (2,708 papers, 5,429 citations, 7 topics)."""
    d = Path(data_dir)
    d.mkdir(parents=True, exist_ok=True)
    content_p, cites_p = d / "cora.content", d / "cora.cites"
    base_url = "https://raw.githubusercontent.com/tkipf/pygcn/master/data/cora"
    for fname, dest in [("cora.content", content_p), ("cora.cites", cites_p)]:
        if not dest.exists() or dest.stat().st_size < 1000:
            urllib.request.urlretrieve(f"{base_url}/{fname}", dest)

    paper_ids, feats, topics = [], [], []
    for line in content_p.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split("\t")
        paper_ids.append(parts[0])
        feats.append([float(x) for x in parts[1:-1]])
        topics.append(parts[-1])

    class_names = sorted(set(topics))
    c2i = {c: i for i, c in enumerate(class_names)}
    y_np = np.array([c2i[t] for t in topics], dtype=np.int32)
    X_np = np.array(feats, dtype=np.float32)
    X_np = X_np / np.maximum(X_np.sum(axis=1, keepdims=True), 1.0)

    id_set = set(paper_ids)
    edges = []
    for line in cites_p.read_text(encoding="utf-8").splitlines():
        u, v = line.strip().split("\t")
        if u in id_set and v in id_set and u != v:
            edges.append((u, v))

    g = Graph(
        num_nodes=len(paper_ids),
        edges=edges,
        node_features=X_np,
        node_ids=paper_ids,
        node_metadata=[{"paper_id": pid, "topic": top} for pid, top in zip(paper_ids, topics)],
        directed=False,
        name="Cora AI Citations",
    )
    return g, vnp.array(y_np, dtype=np.int32), class_names
