import numpy as np
import pytest

from adaptive_biorag.retrieval.dense import DenseRetriever, Embedder, HashingEmbedder, build_embedder


def test_hashing_embedder_deterministic_unit_norm():
    e = HashingEmbedder(128)
    a, b = e.encode_queries(["kernase-7 inhibitor"]), e.encode_queries(["kernase-7 inhibitor"])
    assert np.allclose(a, b) and abs(np.linalg.norm(a[0]) - 1) < 1e-5 and a.shape == (1, 128)
    assert e.encode_passages([]).shape == (0, 128)


def test_dense_retrieval_interface_and_cache(passages, tmp_path):
    d = DenseRetriever(passages, HashingEmbedder(256), tmp_path, "h1")
    res = d.retrieve("valtrexin kernase-7 inhibitor", 2)
    assert res[0].passage_id == "B::p0" and res[0].retriever == "dense" and res[0].rank == 1
    assert list(tmp_path.glob("dense_hashing_*.npy"))
    d2 = DenseRetriever(passages, HashingEmbedder(256), tmp_path, "h1")
    assert np.allclose(d.matrix, d2.matrix) and d.retrieve("", 3) == []


def test_pluggable_embedder(passages):
    class Fake(Embedder):
        backend, model_name = "fake", "f"
        def encode_queries(self, t): return np.tile(np.array([[1.0, 0.0]], np.float32), (len(t), 1))
        def encode_passages(self, t): return np.array([[1.0, 0.0] if "Valtrexin" in x else [0.0, 1.0] for x in t], np.float32)
    assert DenseRetriever(passages, Fake()).retrieve("q", 5)[0].passage_id == "B::p0"


def test_missing_sentence_transformers_gives_clear_error(cfg):
    pytest.importorskip("numpy")
    try:
        import sentence_transformers  # noqa: F401
        pytest.skip("sentence-transformers installed")
    except ImportError:
        cfg["retrieval"]["dense_backend"] = "sentence_transformers"
        with pytest.raises(RuntimeError, match="sentence-transformers"):
            build_embedder(cfg)
