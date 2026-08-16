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


# --- twilight wash: one blend must equal the stack it replaces ---

@pytest.mark.parametrize("tint", [210, 235, 250])
@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_multiply_pow_matches_stacked_multiplies(tint, k):
    # _overlay_night paints k nested bands into one layer instead of blending k times.
    # The cumulative tint must reproduce what k stacked ImageChops.multiply passes did,
    # or twilight steps shift. Pillow rounds each pass, so allow that drift.
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


def _overlay_night_stacked(base, dt, theme, bands, alpha, proj):
    """The pre-collapse overlay: one full-canvas layer and one blend per band.

    Kept here as the reference the single-blend version must reproduce. Nesting order
    is the easy thing to get wrong — the lit side nests the opposite way from the dark
    side — and a picture that is merely plausible would hide it.
    """
    from PIL import Image, ImageChops, ImageDraw
    from worldtime import sun

    w, h = base.size
    sublat, sublon = sun.subsolar_point(dt)
    elevations = render.TWILIGHT_ELEVATIONS if bands else (0.0,)

    def stack(day_side, base_color, tint, op):
        nonlocal base
        for elev in elevations:
            layer = Image.new("RGB", (w, h), base_color)
            ImageDraw.Draw(layer).polygon(
                render._terminator_polygon(elev, sublat, sublon, proj, w, h,
                                           day_side=day_side),
                fill=tint,
            )
            base = render._blend_region(base, layer, op)

    dw = theme.get("day_wash")
    if dw:
        a = dw[3] if len(dw) > 3 else 255
        stack(True, (0, 0, 0), tuple(round(c * a / 255) for c in dw[:3]), ImageChops.screen)
    night = theme.get("night")
    if alpha > 0 and night:
        t = alpha / 255.0
        stack(False, (255, 255, 255),
              tuple(round(255 - (255 - c) * t) for c in night), ImageChops.multiply)
    return base


@pytest.mark.parametrize("month", [6, 12])  # night south of the terminator, then north
@pytest.mark.parametrize("darkness", ["subtle", "dramatic"])
def test_collapsed_twilight_wash_matches_the_stacked_one(month, darkness):
    from PIL import Image, ImageChops
    from worldtime import themes

    dt = datetime(2026, month, 21, 9, tzinfo=timezone.utc)
    th = themes.load_theme("modus")
    alpha = render.DARKNESS_ALPHA[darkness]
    w, h = 400, 250
    proj = render._vector_projection(w, h)
    # A gradient base, so an error anywhere in the tone range shows up.
    src = Image.linear_gradient("L").resize((w, h)).convert("RGBA")

    got = render._overlay_night(src.copy(), dt, th, True, alpha, proj)
    want = _overlay_night_stacked(src.copy(), dt, th, True, alpha, proj)

    diff = ImageChops.difference(got.convert("RGB"), want.convert("RGB"))
    worst = max(hi for _lo, hi in diff.getextrema())
    # Pillow rounds every blend, so four stacked passes drift from the exact wash more
    # than one pass does; the collapsed version is the closer of the two. Anything past
    # a few levels means the bands are nested or counted wrongly, not rounded away.
    assert worst <= 3, f"collapsed wash differs from the stacked one by {worst}/255"


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
    # The whole point: on a whole-hour zone the label is stable between ticks, so the
    # wallpaper only differs by the terminator moving.
    from zoneinfo import ZoneInfo
    vienna = ZoneInfo("Europe/Vienna")
    labels = {
        render._fmt_time(datetime(2026, 8, 16, 12, m, tzinfo=timezone.utc).astimezone(vienna),
                         "hour")
        for m in range(0, 60, 7)
    }
    assert labels == {"14"}
