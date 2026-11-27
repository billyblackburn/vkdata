"""vkgraph.graph: Pure-Vulkan GPU Graph Database, PageRank, BFS, Label Propagation, & GNN Adjacency."""

from __future__ import annotations

import html as _html
import json
from collections import deque
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import vknumpy as vnp


class _GraphSummary(str):
    def __new__(cls, text: str, html_table: str) -> "_GraphSummary":
        obj = super().__new__(cls, text)
        obj._html = html_table
        return obj

    def _repr_html_(self) -> str:
        return self._html


class Graph:
    """In-VRAM Vulkan GPU Graph storing nodes, edges, optional node features, and metadata.

    Examples
    --------
    >>> g = Graph.from_edge_list([(0, 1), (1, 2), (2, 0)])
    >>> pr = g.pagerank()
    >>> path = g.shortest_path(0, 2)
    """

    def __init__(
        self,
        num_nodes: int,
        edges: Sequence[Tuple[int, int]],
        *,
        weights: Optional[Sequence[float]] = None,
        node_features: Optional[Any] = None,
        node_ids: Optional[Sequence[Any]] = None,
        node_metadata: Optional[Sequence[Any]] = None,
        directed: bool = False,
        name: str = "graph",
    ) -> None:
        self.num_nodes = int(num_nodes)
        self.directed = bool(directed)
        self.name = str(name)

        self.node_ids: List[Any] = list(node_ids) if node_ids is not None else list(range(self.num_nodes))
        self.id_to_idx: Dict[Any, int] = {nid: i for i, nid in enumerate(self.node_ids)}
        self.node_metadata: List[Any] = (
            list(node_metadata) if node_metadata is not None else [None] * self.num_nodes
        )

        # Normalize edge list to integer indices
        src_list, dst_list, w_list = [], [], []
        w_in = list(weights) if weights is not None else [1.0] * len(edges)
        self._adj_list: List[List[int]] = [[] for _ in range(self.num_nodes)]

        for (u_raw, v_raw), w in zip(edges, w_in):
            u = self.id_to_idx[u_raw] if u_raw in self.id_to_idx else int(u_raw)
            v = self.id_to_idx[v_raw] if v_raw in self.id_to_idx else int(v_raw)
            if not (0 <= u < self.num_nodes and 0 <= v < self.num_nodes):
                continue
            src_list.append(u)
            dst_list.append(v)
            w_list.append(float(w))
            self._adj_list[u].append(v)
            if not self.directed and u != v:
                self._adj_list[v].append(u)

        self.src = np.array(src_list, dtype=np.int32)
        self.dst = np.array(dst_list, dtype=np.int32)
        self.weights = np.array(w_list, dtype=np.float32)

        if node_features is not None:
            if isinstance(node_features, vnp.ndarray):
                self.features: Optional[vnp.ndarray] = node_features.astype(np.float32)
            elif hasattr(node_features, "_columns") and hasattr(node_features, "columns"):
                self.features = vnp.column_stack([node_features[c]._data for c in node_features.columns]).astype(np.float32)
            else:
                self.features = vnp.array(np.asarray(node_features, dtype=np.float32), dtype=np.float32)
        else:
            self.features = None

        self._adj_gpu: Optional[vnp.ndarray] = None

    @property
    def num_edges(self) -> int:
        return int(len(self.src))

    def __len__(self) -> int:
        return self.num_nodes

    def _resolve_node(self, node: Any) -> int:
        if node in self.id_to_idx:
            return self.id_to_idx[node]
        idx = int(node)
        if not (0 <= idx < self.num_nodes):
            raise KeyError(f"Node {node!r} not in graph")
        return idx

    @classmethod
    def from_edge_list(
        cls,
        edges: Sequence[Tuple[Any, Any]],
        *,
        num_nodes: Optional[int] = None,
        weights: Optional[Sequence[float]] = None,
        node_features: Optional[Any] = None,
        node_ids: Optional[Sequence[Any]] = None,
        node_metadata: Optional[Sequence[Any]] = None,
        directed: bool = False,
        name: str = "graph",
    ) -> "Graph":
        """Build a GPU Graph from a list of ``(u, v)`` pairs."""
        if node_ids is None and num_nodes is None:
            seen: Dict[Any, int] = {}
            for u, v in edges:
                seen.setdefault(u, len(seen))
                seen.setdefault(v, len(seen))
            node_ids = list(seen.keys())
            num_nodes = len(node_ids)
        elif num_nodes is None:
            assert node_ids is not None
            num_nodes = len(node_ids)
        return cls(
            num_nodes=num_nodes,
            edges=edges,
            weights=weights,
            node_features=node_features,
            node_ids=node_ids,
            node_metadata=node_metadata,
            directed=directed,
            name=name,
        )

    @classmethod
    def from_dataframes(
        cls,
        edges_df: Any,
        *,
        src_col: str = "src",
        dst_col: str = "dst",
        weight_col: Optional[str] = None,
        nodes_df: Optional[Any] = None,
        id_col: str = "id",
        directed: bool = False,
        name: str = "graph",
    ) -> "Graph":
        """Build a GPU Graph directly from ``vkpandas.DataFrame`` tables (plug-and-play!)."""
        srcs = edges_df[src_col].to_list()
        dsts = edges_df[dst_col].to_list()
        edges = list(zip(srcs, dsts))
        weights = edges_df[weight_col].to_list() if weight_col is not None else None
        node_ids = nodes_df[id_col].to_list() if nodes_df is not None else None
        return cls.from_edge_list(
            edges, weights=weights, node_ids=node_ids, directed=directed, name=name
        )

    def adjacency_matrix(self, *, add_self_loops: bool = False) -> vnp.ndarray:
        """Return the ``(N, N)`` adjacency matrix in Vulkan GPU VRAM."""
        if not add_self_loops and self._adj_gpu is not None:
            return self._adj_gpu
        n = self.num_nodes
        A = np.zeros((n, n), dtype=np.float32)
        if len(self.src) > 0:
            A[self.src, self.dst] = self.weights
            if not self.directed:
                A[self.dst, self.src] = self.weights
        if add_self_loops:
            np.fill_diagonal(A, 1.0)
            return vnp.array(A, dtype=np.float32)
        self._adj_gpu = vnp.array(A, dtype=np.float32)
        return self._adj_gpu

    def normalized_adjacency(self, *, add_self_loops: bool = True) -> vnp.ndarray:
        """Return symmetric normalized adjacency ``D^(-1/2) A_tilde D^(-1/2)`` in GPU VRAM for GCNs."""
        A = self.adjacency_matrix(add_self_loops=add_self_loops)
        deg = vnp.sum(A, axis=1)
        inv_sqrt = vnp.where(deg > 0.0, 1.0 / vnp.sqrt(vnp.maximum(deg, 1e-12)), 0.0)
        # Scale rows and columns: (inv_sqrt[:, None] * A) * inv_sqrt[None, :]
        return (inv_sqrt.reshape(-1, 1) * A) * inv_sqrt.reshape(1, -1)

    def degree(self) -> vnp.ndarray:
        """Compute every node's degree (number of connections) on Vulkan GPU."""
        return vnp.sum(self.adjacency_matrix(add_self_loops=False), axis=1)

    def neighbors(self, node: Any) -> List[int]:
        """Return direct 1-hop neighbor indices of ``node``."""
        return list(dict.fromkeys(self._adj_list[self._resolve_node(node)]))

    def k_hop_neighbors(self, node: Any, *, k: int = 2) -> Dict[int, int]:
        """Return ``{node_idx: hop_distance}`` for all nodes within ``k`` hops (ideal for GraphRAG!)."""
        start = self._resolve_node(node)
        visited: Dict[int, int] = {start: 0}
        q: deque = deque([start])
        while q:
            cur = q.popleft()
            d = visited[cur]
            if d >= int(k):
                continue
            for nxt in self._adj_list[cur]:
                if nxt not in visited:
                    visited[nxt] = d + 1
                    q.append(nxt)
        return visited

    def shortest_path(self, start: Any, goal: Any) -> List[int]:
        """Find the shortest unweighted path (BFS) between ``start`` and ``goal``."""
        s, g = self._resolve_node(start), self._resolve_node(goal)
        if s == g:
            return [s]
        parent: Dict[int, Optional[int]] = {s: None}
        q: deque = deque([s])
        while q:
            cur = q.popleft()
            for nxt in self._adj_list[cur]:
                if nxt not in parent:
                    parent[nxt] = cur
                    if nxt == g:
                        path = []
                        p: Optional[int] = g
                        while p is not None:
                            path.append(p)
                            p = parent[p]
                        return path[::-1]
                    q.append(nxt)
        return []

    def pagerank(self, *, damping: float = 0.85, max_iter: int = 50, tol: float = 1e-6) -> vnp.ndarray:
        """Compute PageRank node importance scores via Vulkan GPU power iteration."""
        n = self.num_nodes
        if n == 0:
            return vnp.zeros((0,), dtype=np.float32)
        A = self.adjacency_matrix(add_self_loops=False)
        out_deg = vnp.sum(A, axis=1)
        # Transition matrix M (column-stochastic: M[j, i] = A[i, j] / out_deg[i])
        inv_deg = vnp.where(out_deg > 0.0, 1.0 / vnp.maximum(out_deg, 1e-12), 0.0)
        M = (A * inv_deg.reshape(-1, 1)).T.contiguous()

        pr = vnp.full((n, 1), 1.0 / n, dtype=np.float32)
        teleport = float((1.0 - damping) / n)
        d_f = np.float32(damping)

        for _ in range(int(max_iter)):
            pr_next = (M @ pr) * d_f + teleport
            # Re-normalize to handle dangling nodes cleanly
            pr_next = pr_next / vnp.sum(pr_next)
            diff = float(vnp.sum(vnp.abs(pr_next - pr)))
            pr = pr_next
            if diff < tol:
                break
        return pr.reshape(-1)

    def label_propagation(
        self,
        seed_labels: Any,
        *,
        num_classes: Optional[int] = None,
        alpha: float = 0.85,
        max_iter: int = 40,
    ) -> vnp.ndarray:
        """GPU Semi-Supervised Label Propagation (nodes with label ``-1`` inherit labels from neighbors!)."""
        labels_np = seed_labels.to_numpy() if hasattr(seed_labels, "to_numpy") else np.asarray(seed_labels)
        labels_np = labels_np.astype(np.int32).ravel()
        known = labels_np >= 0
        k = int(num_classes) if num_classes is not None else (int(labels_np[known].max()) + 1 if known.any() else 1)

        Y0_np = np.zeros((self.num_nodes, k), dtype=np.float32)
        known_idx = np.nonzero(known)[0]
        Y0_np[known_idx, labels_np[known_idx]] = 1.0

        S = self.normalized_adjacency(add_self_loops=False)
        Y0 = vnp.array(Y0_np, dtype=np.float32)
        Y = Y0
        a_f = np.float32(alpha)
        one_m = np.float32(1.0 - alpha)
        for _ in range(int(max_iter)):
            Y = (S @ Y) * a_f + Y0 * one_m
        return vnp.argmax(Y, axis=1)

    def describe(self) -> _GraphSummary:
        """Return a readable summary of nodes, edges, degree, and density (renders as HTML in Jupyter)."""
        deg = self.degree()
        avg_deg = float(vnp.mean(deg)) if self.num_nodes else 0.0
        max_deg = int(float(vnp.max(deg))) if self.num_nodes else 0
        max_possible = self.num_nodes * (self.num_nodes - 1) / (1.0 if self.directed else 2.0)
        density = self.num_edges / max_possible if max_possible > 0 else 0.0
        feat_str = f"{self.features.shape[1]}-D float32" if self.features is not None else "None"
        kind = "Directed" if self.directed else "Undirected"

        txt = (
            f"vkgraph.Graph '{self.name}' ({kind}) | "
            f"nodes={self.num_nodes:,}, edges={self.num_edges:,}, "
            f"avg_deg={avg_deg:.2f}, max_deg={max_deg}, density={density:.4f}, features={feat_str}"
        )
        html_t = (
            f"<div style='overflow-x:auto;'>"
            f"<div style='font-size:12px;opacity:0.8;margin-bottom:4px;'>"
            f"<strong>vkgraph.Graph</strong> &middot; {_html.escape(self.name)} ({kind}, Vulkan GPU)</div>"
            f"<table style='border-collapse:collapse;font-family:ui-monospace,monospace;font-size:12.5px;'>"
            f"<thead><tr style='border-bottom:2px solid #c792ea;'>"
            f"<th style='padding:6px 10px;'>Nodes</th><th style='padding:6px 10px;'>Edges</th>"
            f"<th style='padding:6px 10px;'>Avg Degree</th><th style='padding:6px 10px;'>Max Degree</th>"
            f"<th style='padding:6px 10px;'>Density</th><th style='padding:6px 10px;'>Node Features</th>"
            f"</tr></thead><tbody><tr>"
            f"<td style='padding:6px 10px;text-align:right;'>{self.num_nodes:,}</td>"
            f"<td style='padding:6px 10px;text-align:right;'>{self.num_edges:,}</td>"
            f"<td style='padding:6px 10px;text-align:right;'>{avg_deg:.2f}</td>"
            f"<td style='padding:6px 10px;text-align:right;'>{max_deg:,}</td>"
            f"<td style='padding:6px 10px;text-align:right;'>{density:.4f}</td>"
            f"<td style='padding:6px 10px;text-align:center;'>{feat_str}</td>"
            f"</tr></tbody></table></div>"
        )
        return _GraphSummary(txt, html_t)

    def save(self, path: Union[str, Path]) -> Path:
        """Persist graph edges, weights, features (.npz), and metadata (.meta.json) to disk."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        feat_np = np.zeros((0, 0), dtype=np.float32) if self.features is None else self.features.to_numpy()
        np.savez_compressed(p, src=self.src, dst=self.dst, weights=self.weights, features=feat_np)
        p.with_suffix(".meta.json").write_text(
            json.dumps(
                {
                    "num_nodes": self.num_nodes,
                    "directed": self.directed,
                    "name": self.name,
                    "node_ids": self.node_ids,
                    "node_metadata": self.node_metadata,
                }
            ),
            encoding="utf-8",
        )
        return p

    @classmethod
    def load(cls, path: Union[str, Path]) -> "Graph":
        """Reload a saved ``Graph`` directly into Vulkan GPU VRAM."""
        p = Path(path)
        meta = json.loads(p.with_suffix(".meta.json").read_text(encoding="utf-8"))
        data = np.load(p)
        edges = list(zip(data["src"].tolist(), data["dst"].tolist()))
        feats = data["features"] if data["features"].size > 0 else None
        return cls(
            num_nodes=meta["num_nodes"],
            edges=edges,
            weights=data["weights"].tolist(),
            node_features=feats,
            node_ids=meta["node_ids"],
            node_metadata=meta["node_metadata"],
            directed=meta["directed"],
            name=meta["name"],
        )
