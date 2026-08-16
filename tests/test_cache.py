"""The on-disk cache for the vector base map (render._vector_base).

The cache must be invisible in the output (hit, miss and uncached renders byte-equal),
self-repairing when a cache file is damaged, keyed so different inputs get different
files, and bounded in size.
"""
from datetime import datetime, timezone

from worldtime import render

CITIES = [
    {"name": "Vienna", "lat": 48.21, "lon": 16.37, "tz": "Europe/Vienna", "home": True},
    {"name": "Tokyo", "lat": 35.68, "lon": 139.69, "tz": "Asia/Tokyo"},
]
DT = datetime(2026, 8, 16, 9, 0, tzinfo=timezone.utc)


def _render(**kw):
    kw.setdefault("out_size", (320, 200))
    return render.render(CITIES, dt=DT, map_style="vector", logo=False, **kw)


def _cache_files(tmp_path):
    return sorted(p for p in tmp_path.iterdir()
                  if p.name.startswith("base-") and p.name.endswith(".png"))


def test_cached_renders_match_uncached(tmp_path):
    plain = _render()
    cold = _render(cache_dir=str(tmp_path))
    assert len(_cache_files(tmp_path)) == 1
    warm = _render(cache_dir=str(tmp_path))
    assert len(_cache_files(tmp_path)) == 1
    assert cold.tobytes() == plain.tobytes()
    assert warm.tobytes() == plain.tobytes()


def test_damaged_cache_file_is_rebuilt(tmp_path):
    plain = _render()
    _render(cache_dir=str(tmp_path))
    (path,) = _cache_files(tmp_path)
    path.write_bytes(path.read_bytes()[: 100])  # a watcher-style truncated PNG
    repaired = _render(cache_dir=str(tmp_path))
    assert repaired.tobytes() == plain.tobytes()
    (path,) = _cache_files(tmp_path)
    assert path.read_bytes()[:4] == b"\x89PNG" and path.stat().st_size > 100


def test_key_separates_differing_inputs(tmp_path):
    _render(cache_dir=str(tmp_path))
    _render(cache_dir=str(tmp_path), theme="gruvbox")
    _render(cache_dir=str(tmp_path), out_size=(200, 125))
    assert len(_cache_files(tmp_path)) == 3


def test_cache_is_pruned_to_a_few_entries(tmp_path):
    for i in range(render._BASE_CACHE_KEEP + 2):
        (tmp_path / f"base-{i:016d}.png").write_bytes(b"old")
    _render(cache_dir=str(tmp_path))
    files = _cache_files(tmp_path)
    assert len(files) == render._BASE_CACHE_KEEP
    assert any(f.stat().st_size > 100 for f in files)  # the fresh entry survived


def test_unwritable_cache_degrades_to_an_uncached_render(tmp_path, capsys):
    plain = _render()
    ro = tmp_path / "ro"
    ro.mkdir()
    ro.chmod(0o500)
    try:
        img = _render(cache_dir=str(ro / "greyline"))
    finally:
        ro.chmod(0o700)
    assert img.tobytes() == plain.tobytes()
    assert "cache write failed" in capsys.readouterr().err
