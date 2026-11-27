"""Unit tests for vkgraph (Vulkan GPU Graph, PageRank, BFS, Label Propagation, GCN Adjacency, & Viz)."""

import numpy as np
import vkpandas as vpd

import vkgraph


def test_karate_club_pagerank_bfs_khop_and_label_propagation():
    g, factions = vkgraph.karate_club()
    assert g.num_nodes == 34 and g.num_edges == 78

    # GPU PageRank: top-2 most influential leaders must be node 33 (Officer) and node 0 (Mr. Hi)!
    pr = g.pagerank().to_numpy()
    np.testing.assert_allclose(pr.sum(), 1.0, atol=1e-4)
    top2 = set(np.argsort(pr)[-2:].tolist())
    assert top2 == {0, 33}

    # BFS shortest path & k-hop GraphRAG neighborhood
    path = g.shortest_path(16, 26)
    assert path[0] == 16 and path[-1] == 26 and len(path) >= 4
    hops = g.k_hop_neighbors(0, k=2)
    assert hops[0] == 0 and 1 in hops and hops[1] == 1

    # Semi-supervised GPU Label Propagation from just 2 labeled leaders (0 & 33) recovers >90% of the club!
    seeds = np.full(34, -1, dtype=np.int32)
    seeds[0] = 0
    seeds[33] = 1
    pred = g.label_propagation(seeds, num_classes=2).to_numpy()
    acc = float((pred == factions).mean())
    assert acc >= 0.90

    # Symmetric normalized adjacency for GCNs
    A_hat = g.normalized_adjacency(add_self_loops=True).to_numpy()
    assert A_hat.shape == (34, 34)
    np.testing.assert_allclose(A_hat, A_hat.T, atol=1e-5)

    # Summary HTML + SynthWave graph render
    desc = g.describe()
    assert "34" in str(desc) and "<table" in desc._repr_html_()
    im = vkgraph.render_graph(g, labels=factions, node_sizes=pr, highlight_path=path, class_names=["Mr. Hi", "Officer"])
    assert im.size == (860, 520)


def test_graph_from_dataframes_and_persistence(tmp_path):
    edges_df = vpd.DataFrame({"src": ["A", "B", "C"], "dst": ["B", "C", "A"]})
    g = vkgraph.Graph.from_dataframes(edges_df, src_col="src", dst_col="dst", name="triangle")
    assert g.num_nodes == 3 and g.num_edges == 3
    assert g.shortest_path("A", "C") == [0, 2]

    # Save & load roundtrip
    save_p = tmp_path / "tri.npz"
    g.save(save_p)
    g2 = vkgraph.Graph.load(save_p)
    assert g2.num_nodes == 3 and g2.num_edges == 3 and g2.node_ids == ["A", "B", "C"]
