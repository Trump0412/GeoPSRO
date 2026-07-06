from __future__ import annotations

from geopsro4d.data.schema import normalize_sample


def test_normalize_sample_aliases() -> None:
    sample = normalize_sample(
        {
            "clip_id": "c1",
            "source_dataset": "spar",
            "frame_paths": ["a.jpg"],
            "question": "Where?",
            "answer": "left",
        }
    )
    assert sample.sample_id == "c1"
    assert sample.dataset == "spar"
    assert sample.media_paths == ("a.jpg",)
