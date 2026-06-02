from __future__ import annotations

import json
import tempfile
from pathlib import Path

from embedopt.evaluation.beir import load_beir_dataset_local


def test_load_beir_dataset_local_round_trip() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "qrels").mkdir()
        with (root / "corpus.jsonl").open("w", encoding="utf-8") as f:
            for cid, (title, text) in enumerate(
                [
                    ("Cats", "Cats are small carnivorous mammals."),
                    ("Dogs", "Dogs are domesticated descendants of wolves."),
                    ("Stars", "Stars are luminous balls of plasma."),
                ]
            ):
                f.write(json.dumps({"_id": str(cid), "title": title, "text": text}) + "\n")
        with (root / "queries.jsonl").open("w", encoding="utf-8") as f:
            f.write(json.dumps({"_id": "q0", "text": "what is a cat"}) + "\n")
            f.write(json.dumps({"_id": "q1", "text": "what is a star"}) + "\n")
        with (root / "qrels" / "test.tsv").open("w", encoding="utf-8") as f:
            f.write("query-id\tcorpus-id\tscore\n")
            f.write("q0\t0\t1\n")
            f.write("q1\t2\t1\n")
        ds = load_beir_dataset_local(root, name="toy")
    assert ds.name == "toy"
    assert len(ds.corpus) == 3
    assert len(ds.queries) == 2
    assert ds.qrels[0] == {0: 1.0}
    assert ds.qrels[1] == {2: 1.0}
