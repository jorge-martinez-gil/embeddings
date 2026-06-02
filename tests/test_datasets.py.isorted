from __future__ import annotations

from embedopt.evaluation.datasets import make_synthetic_retrieval, make_synthetic_sts


def test_sts_dataset_is_deterministic() -> None:
    a = make_synthetic_sts(n_pairs=20, seed=3)
    b = make_synthetic_sts(n_pairs=20, seed=3)
    assert list(a.sentences_a) == list(b.sentences_a)
    assert list(a.scores) == list(b.scores)


def test_sts_dataset_shape() -> None:
    d = make_synthetic_sts(n_pairs=50, seed=0)
    assert len(d.sentences_a) == 50
    assert len(d.sentences_b) == 50
    assert len(d.scores) == 50
    assert all(0.0 <= s <= 5.0 for s in d.scores)


def test_retrieval_dataset_qrels_well_formed() -> None:
    d = make_synthetic_retrieval(n_queries_per_topic=2, seed=0)
    assert len(d.queries) == 4 * 2  # four topics, two queries each
    for qid, qrel in d.qrels.items():
        assert qid in range(len(d.queries))
        assert any(r > 0 for r in qrel.values())
        for doc_id in qrel:
            assert 0 <= doc_id < len(d.corpus)
