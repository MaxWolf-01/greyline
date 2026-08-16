"""End-to-end smoke: both map styles produce an RGB image of the requested size."""
import os
from datetime import datetime, timezone

import pytest

from worldtime import render

CITIES = [
    {"name": "London", "lat": 51.51, "lon": -0.13, "tz": "Europe/London", "home": True},
    {"name": "Tokyo", "lat": 35.68, "lon": 139.69, "tz": "Asia/Tokyo", "home": False},
]
DT = datetime(2024, 6, 20, 9, 30, tzinfo=timezone.utc)


@pytest.mark.parametrize("style", ["raster", "vector"])
def test_render_returns_rgb_at_size(style):
    if style == "raster" and not os.path.isfile(render.BASE_1400):
        pytest.skip("raster map artwork not bundled (IBM/Lenovo art, see NOTICE)")
    img = render.render(CITIES, dt=DT, out_size=(480, 300), map_style=style)
    assert img.size == (480, 300)
    assert img.mode == "RGB"


@pytest.mark.parametrize(
    "tz, lon, month, expected",
    [
        # DST-observing zones must highlight the *standard* (geographic) column all year,
        # not the DST-shifted one — regression for issue #14.
        ("Europe/London", -0.13, 1, 0.0),    # GMT (winter)
        ("Europe/London", -0.13, 7, 0.0),    # BST → must still be 0, not +1
        ("America/New_York", -74.0, 1, -5.0), # EST
        ("America/New_York", -74.0, 7, -5.0), # EDT → must still be -5, not -4
        # Southern hemisphere: DST is in Jan, so the naive min-of-two heuristic would fail.
        ("Australia/Sydney", 151.2, 1, 10.0), # AEDT → must be +10, not +11
        ("Australia/Sydney", 151.2, 7, 10.0), # AEST
    ],
)
def test_home_column_uses_standard_offset(monkeypatch, tz, lon, month, expected):
    captured = {}
    real_build_base = render.vectormap.build_base

    def spy(*args, **kwargs):
        captured["home_offset"] = kwargs.get("home_offset")
        return real_build_base(*args, **kwargs)

    monkeypatch.setattr(render.vectormap, "build_base", spy)
    cities = [{"name": "Home", "lat": 0.0, "lon": lon, "tz": tz, "home": True}]
    dt = datetime(2026, month, 15, 12, tzinfo=timezone.utc)
    render.render(cities, dt=dt, out_size=(320, 200), map_style="vector")
    assert captured["home_offset"] == expected


# --- twilight wash ---

@pytest.mark.parametrize("tint", [210, 235, 250])
@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_multiply_pow_matches_stacked_multiplies(tint, k):
    # _overlay_night blends once, tinting each pixel by how many bands cover it, rather
    # than blending once per band. The cumulative tint must reproduce what k stacked
    # ImageChops.multiply passes did, or twilight steps shift. Pillow rounds each pass,
    # so allow that drift.
    for base in range(0, 256, 7):
        stacked = base
        for _ in range(k):
            stacked = round(stacked * tint / 255)
        collapsed = round(base * render._multiply_pow((tint,), k)[0] / 255)
        assert abs(collapsed - stacked) <= 3


@pytest.mark.parametrize("tint", [8, 24, 60])
@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_screen_pow_matches_stacked_screens(tint, k):
    for base in range(0, 256, 7):
        stacked = base
        for _ in range(k):
            stacked = 255 - round((255 - stacked) * (255 - tint) / 255)
        collapsed = 255 - round((255 - base) * (255 - render._screen_pow((tint,), k)[0]) / 255)
        assert abs(collapsed - stacked) <= 3


def _true_band_depth(w, h, proj, sublat, sublon, elevations):
    """Bands below each pixel, straight from the solar-elevation equation.

    No projection tricks, no polygons — evaluate

        sin(elev) = sin(lat)sin(dec) + cos(lat)cos(dec)cos(lon - sublon)

    at each pixel and count the thresholds it falls under. Slow, and the definition
    the renderer's row-of-arcs version has to reproduce.
    """
    import math

    from PIL import Image

    dec = math.radians(sublat)
    sin_dec, cos_dec = math.sin(dec), math.cos(dec)
    thresholds = sorted(math.sin(math.radians(e)) for e in elevations)
    out = Image.new("L", (w, h), 0)
    px = out.load()
    for y in range(h):
        lat = math.radians(proj.y_to_lat(y + 0.5))
        sin_lat, cos_lat = math.sin(lat), math.cos(lat)
        for x in range(w):
            hour_angle = math.radians(proj.x_to_lon(x + 0.5) - sublon)
            v = sin_lat * sin_dec + cos_lat * cos_dec * math.cos(hour_angle)
            px[x, y] = sum(1 for t in thresholds if v < t)
    return out


# Dates chosen by subsolar latitude, which is what the band geometry turns on. The
# solstices alone are not enough: with the sun south of the equator the dark region
# reaches the opposite pole, and within 18 degrees of it the deep bands reach neither,
# so a model built on a single boundary curve closed against one edge of the canvas
# gets those wrong while looking right in June.
@pytest.mark.parametrize(
    "month, day",
    [
        (6, 21),   # sublat +23.4
        (12, 21),  # sublat -23.4
        (2, 20),   # sublat ~-11
        (3, 21),   # sublat ~0, the degenerate equinox
        (9, 23),   # sublat ~0 going the other way
        (10, 15),  # sublat ~-8
    ],
)
@pytest.mark.parametrize("style", ["vector", "raster"])
def test_band_depth_matches_the_solar_elevation_equation(month, day, style):
    from PIL import ImageChops

    w, h = 200, 125
    proj = (render._vector_projection(w, h) if style == "vector"
            else render._raster_projection(w, h, (0.5, 1.0))[0])
    dt = datetime(2026, month, day, 9, tzinfo=timezone.utc)
    sublat, sublon = render.sun.subsolar_point(dt)

    got = render._band_depth(w, h, proj, sublat, sublon, render.TWILIGHT_ELEVATIONS)
    want = _true_band_depth(w, h, proj, sublat, sublon, render.TWILIGHT_ELEVATIONS)

    diff = ImageChops.difference(got, want)
    worst = max(diff.get_flattened_data())
    wrong = sum(1 for p in diff.get_flattened_data() if p)
    # Filling whole pixels can only place a boundary to within one of them, so a band
    # edge lands one step out either way. A deeper error means the geometry is wrong.
    assert worst <= 1, f"depth off by {worst} bands"
    assert wrong < 0.05 * w * h, f"{wrong} of {w * h} pixels differ — more than edges"


@pytest.mark.parametrize(
    "elevation, along, across, expected",
    [
        (0.0, 0.0, 1.0, 90.0),    # equator at equinox: half the row is dark
        (0.0, 0.9, 0.1, 0.0),     # midnight sun: the shadow never reaches this row
        (0.0, -0.9, 0.1, 180.0),  # polar night: the whole row is dark
        (0.0, 0.5, 0.0, 0.0),     # a pole, lit
        (0.0, -0.5, 0.0, 180.0),  # a pole, dark
    ],
)
def test_arc_half_width_degenerate_rows(elevation, along, across, expected):
    import math
    got = render._arc_half_width(math.sin(math.radians(elevation)), along, across)
    assert abs(got - expected) < 1e-9


def test_unknown_theme_falls_back():
    img = render.render(CITIES, dt=DT, out_size=(320, 200), theme="does-not-exist")
    assert img.size == (320, 200)


def test_hex_parsing_and_bad_values():
    assert render._hex("#e64553") == (230, 69, 83)
    assert render._hex("990000") == (153, 0, 0)  # no leading '#'
    assert render._hex("#fff") == (255, 255, 255)  # #rgb shorthand
    assert render._hex("000000") == (0, 0, 0)  # black is a real colour, not "unset"
    # 8 digits carry a trailing alpha (theme-file format).
    assert render._hex("#58b88034") == (88, 184, 128, 52)
    assert render._hex("11223344") == (17, 34, 51, 68)
    # Unparseable values return None (theme default) instead of raising.
    for bad in (None, 990000, "", "#ggg", "#12345", "#1234567", "notacolor"):
        assert render._hex(bad) is None


def test_logo_scale_shrinks_the_logo():
    from PIL import Image
    th = render.THEMES["dark"]
    full = render._draw_logo(Image.new("RGBA", (1000, 600)), th, render.LOGO_PNG,
                             logo_scale=1.0)
    half = render._draw_logo(Image.new("RGBA", (1000, 600)), th, render.LOGO_PNG,
                             logo_scale=0.5)
    w_full, w_half = full[2] - full[0], half[2] - half[0]
    assert w_half < w_full and abs(w_half - w_full / 2) <= 2


def test_logo_max_height_caps_tall_logos(tmp_path):
    from PIL import Image
    th = render.THEMES["dark"]
    canvas = (1000, 600)
    # A tall/portrait logo would otherwise blow up: at the fixed ~10.4% width its
    # aspect-derived height dwarfs the canvas. The cap bounds it (and keeps aspect).
    tall = tmp_path / "tall.png"
    Image.new("RGBA", (100, 1000), (255, 0, 0, 255)).save(tall)

    uncapped = render._draw_logo(Image.new("RGBA", canvas), th, str(tall))
    capped = render._draw_logo(Image.new("RGBA", canvas), th, str(tall),
                               logo_max_height=0.5)
    h_uncapped = uncapped[3] - uncapped[1]
    h_capped = capped[3] - capped[1]
    assert h_capped <= 600 * 0.5 + 1 < h_uncapped
    # Aspect ratio preserved after capping (tall source stays 1:10).
    w_capped = capped[2] - capped[0]
    assert abs(w_capped - h_capped / 10) <= 1


# --- hour-only labels ---

@pytest.mark.parametrize(
    "tz, expected",
    [
        ("Europe/Vienna", "14"),        # +2:00 — whole hour, minutes dropped
        ("America/New_York", "08"),     # -4:00
        ("Asia/Tokyo", "21"),           # +9:00
        ("Asia/Kolkata", "17:30"),      # +5:30 — must keep its half hour
        ("Asia/Kathmandu", "17:45"),    # +5:45
        ("Pacific/Chatham", "00:45"),   # +12:45, next day
        ("Australia/Adelaide", "21:30"),  # +9:30
    ],
)
def test_hour_format_keeps_minutes_only_where_the_zone_needs_them(tz, expected):
    from zoneinfo import ZoneInfo
    dt = datetime(2026, 8, 16, 12, 0, tzinfo=timezone.utc)
    assert render._fmt_time(dt.astimezone(ZoneInfo(tz)), "hour") == expected


def test_hour_format_does_not_change_within_an_hour():
    # On a whole-hour zone the label is stable between ticks, so consecutive wallpapers
    # differ only by the terminator moving.
    from zoneinfo import ZoneInfo
    vienna = ZoneInfo("Europe/Vienna")
    labels = {
        render._fmt_time(datetime(2026, 8, 16, 12, m, tzinfo=timezone.utc).astimezone(vienna),
                         "hour")
        for m in range(0, 60, 7)
    }
    assert labels == {"14"}


def test_label_placement_keeps_backplates_apart():
    """Two chips whose text boxes clear each other can still collide plate-to-plate:
    the backplate pads beyond the box. Placement must hold the inflated boxes apart
    (the London/Vienna geometry that shipped overlapping)."""
    items = [
        {"is_home": True, "px": 200, "py": 100, "w": 80, "h": 20, "dotr": 4},
        {"is_home": False, "px": 100, "py": 102, "w": 80, "h": 20, "dotr": 4},
    ]
    render._place_labels(items, [], (0, 0, 1000, 500), scale=1.0, inflate=(15, 10))

    def plate(box):
        return (box[0] - 15, box[1] - 10, box[2] + 15, box[3] + 10)

    a, b = items[0]["box"], items[1]["box"]
    assert render._rect_overlap(plate(a), plate(b)) == 0
