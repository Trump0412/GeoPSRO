from __future__ import annotations

from geopsro4d.train.common import make_smoke_cache, make_smoke_model


def test_geometry_modes_shape() -> None:
    _base, wrapper = make_smoke_model(hidden_size=32, num_geo_tokens=4)
    cache = make_smoke_cache(num_frames=2)
    normal = wrapper.geometry_inputs_embeds(cache, mode="normal")
    zero = wrapper.geometry_inputs_embeds(cache, mode="zero")
    assert normal.shape == (1, 4, 32)
    assert zero.shape == (1, 4, 32)
    assert zero.abs().sum().item() == 0.0


def test_shuffle_mode_tokenizes_loaded_geometry() -> None:
    _base, wrapper = make_smoke_model(hidden_size=32, num_geo_tokens=4)
    cache = make_smoke_cache(num_frames=2)
    shuffled = wrapper.geometry_inputs_embeds(cache, mode="shuffle")
    assert shuffled.shape == (1, 4, 32)
    assert shuffled.abs().sum().item() > 0.0
