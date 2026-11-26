"""Unit tests for vkvec (Vulkan GPU VectorIndex, Cosine/L2 search, TF-IDF, & RAG chunking)."""

import numpy as np

import vkvec


def test_normalize_cosine_similarity_and_chunking():
    rng = np.random.default_rng(0)
    a = rng.standard_normal((8, 16)).astype(np.float32)
    b = rng.standard_normal((5, 16)).astype(np.float32)

    a_n = vkvec.normalize(a).to_numpy()
    np.testing.assert_allclose(np.linalg.norm(a_n, axis=1), np.ones(8), atol=1e-5)

    sim_gpu = vkvec.cosine_similarity(a, b).to_numpy()
    a_ref = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_ref = b / np.linalg.norm(b, axis=1, keepdims=True)
    np.testing.assert_allclose(sim_gpu, a_ref @ b_ref.T, atol=1e-4)

    chunks = vkvec.chunk_text(" ".join(f"w{i}" for i in range(55)), chunk_size=20, overlap=5)
    assert len(chunks) == 4
    assert chunks[0].split()[-5:] == chunks[1].split()[:5]


def test_vector_index_cosine_l2_filter_and_persistence(tmp_path):
    docs = [
        "Vulkan GPU compute shaders run matrix math fast on Steam Deck",
        "AMD graphics driver RADV compiles SPIR-V kernels cleanly",
        "Basketball championship playoff game won in overtime by coach",
        "Olympic swimmer breaks world record in freestyle final",
        "Stock market rallies as central bank holds interest rates steady",
        "Oil prices climb while Wall Street investors track inflation",
    ]
    topics = ["Tech", "Tech", "Sports", "Sports", "Finance", "Finance"]

    vec = vkvec.TfidfVectorizer(max_features=200)
    X_gpu = vec.fit_transform(docs)
    assert X_gpu.shape[0] == len(docs)

    # 1. Cosine VectorIndex + metadata filter
    idx = vkvec.VectorIndex(metric="cosine").add(
        X_gpu,
        ids=[f"doc-{i}" for i in range(len(docs))],
        metadata=[{"topic": t, "text": d} for t, d in zip(topics, docs)],
    )
    assert len(idx) == len(docs)

    q_tech = vec.transform(["vulkan shader graphics gpu"])
    hits = idx.search(q_tech, k=2)[0]
    assert hits[0]["metadata"]["topic"] == "Tech"
    assert hits[0]["score"] >= hits[1]["score"]

    # Metadata filter restricts search strictly to Finance
    fin_hits = idx.search(q_tech, k=2, where={"topic": "Finance"})[0]
    assert len(fin_hits) == 2
    assert all(h["metadata"]["topic"] == "Finance" for h in fin_hits)

    # Save & load roundtrip
    save_path = tmp_path / "news_index.npz"
    idx.save(save_path)
    idx2 = vkvec.VectorIndex.load(save_path)
    assert len(idx2) == len(idx)
    hits2 = idx2.search(q_tech, k=2)[0]
    assert [h["id"] for h in hits2] == [h["id"] for h in hits]
    np.testing.assert_allclose([h["score"] for h in hits2], [h["score"] for h in hits], atol=1e-5)

    # 2. L2 VectorIndex exact nearest neighbor
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [10.0, 10.0]], dtype=np.float32)
    l2_idx = vkvec.VectorIndex(metric="l2").add(pts, ids=["origin", "unit_x", "far"])
    l2_hits = l2_idx.search(np.array([[0.9, 0.1]], dtype=np.float32), k=2)[0]
    assert l2_hits[0]["id"] == "unit_x"
    assert l2_hits[0]["score"] < l2_hits[1]["score"]
