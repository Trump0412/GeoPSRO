from __future__ import annotations

from geopsro4d.data.frame_sampler import sample_frame_selection, sample_frames


def test_multi_image_uniform_preserves_order() -> None:
    out = sample_frames(["a.jpg", "b.jpg", "c.jpg", "d.jpg"], 3)
    assert out == ["a.jpg", "c.jpg", "d.jpg"]


def test_short_sequence_repeats_last() -> None:
    sel = sample_frame_selection(["a.jpg", "b.jpg"], 4)
    assert sel.paths == ["a.jpg", "b.jpg", "b.jpg", "b.jpg"]
    assert sel.indices == [0, 1, 1, 1]
