"""Compose the World Time wallpaper image (PORTABLE CORE; deps: Pillow).

Pipeline (see plan): work in the map's 1400x1050 calibration frame for the smooth
day/night + twilight overlays and the home timezone-column highlight; cover-crop the
composited map to the target output size; then draw the city clocks at NATIVE output
resolution so text stays crisp on HiDPI panels.
"""
import hashlib
import math
import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFont, ImageOps

from . import __version__, geo, sun, themes, vectormap
from .themes import _hex  # re-exported here for back-compat (moved to themes.py)

ASSET_DIR = os.path.join(os.path.dirname(__file__), "assets")
BASE_1400 = os.path.join(ASSET_DIR, "world.time.1400x1050.png")
LOGO_PNG = os.path.join(ASSET_DIR, "tux.png")  # default corner logo (see NOTICE for swapping it)

# Twilight boundaries (solar elevation, degrees): terminator + civil/nautical/astro.
# Each is filled as a night-side polygon; stacking translucent layers darkens the
# deeper-night regions progressively.
TWILIGHT_ELEVATIONS = (0.0, -6.0, -12.0, -18.0)

# Per-layer overlay alpha by darkness preset (4 stacked layers at full night).
DARKNESS_ALPHA = {"subtle": 28, "medium": 40, "dramatic": 55}

# Built-in palette snapshot (parsed from worldtime/themes/*.toml). render() goes
# through themes.load_theme() so user themes and [colors] overrides work; this dict
# stays for tests and the web-demo port. "dark" is the pre-0.6 name for modus.
THEMES = themes.builtin_themes()
THEMES["dark"] = THEMES["modus"]

# Font candidates (Aporetic preferred per the repo; DejaVu as the portable fallback).
# The Windows (segoe/arial) and macOS (Helvetica/SFNS) names are found by Pillow's
# own OS-font-dir search; if none resolve, _load_font falls back to load_default().
FONT_CANDIDATES = [
    "Aporetic Sans", "AporeticSans", "Aporetic-Sans",
    "DejaVuSans.ttf", "DejaVu Sans",
    "segoeui.ttf", "arial.ttf",          # Windows
    "Helvetica.ttc", "SFNS.ttf", "Arial.ttf",  # macOS
]
FONT_BOLD_CANDIDATES = [
    "Aporetic Sans Bold", "AporeticSans-Bold",
    "DejaVuSans-Bold.ttf", "DejaVu Sans Bold",
    "segoeuib.ttf", "arialbd.ttf",       # Windows
    "Helvetica.ttc", "SFNS.ttf", "Arial Bold.ttf",  # macOS
]


def _load_font(size, candidates, explicit=None):
    for name in ([explicit] if explicit else []) + candidates:
        if not name:
            continue
        try:
            return ImageFont.truetype(name, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default(size)


class Projection:
    """Maps geographic lon/lat to OUTPUT pixel coordinates (and back, for x and lat).

    `scale` is a sizing factor (relative to the 1400-wide reference) for fonts/dots so
    both map styles look consistent.
    """

    def __init__(self, to_px, x_to_lon, y_to_lat, scale):
        self.to_px = to_px
        self.x_to_lon = x_to_lon
        self.y_to_lat = y_to_lat
        self.scale = scale


def _cover_transform(ref_w, ref_h, out_w, out_h, anchor):
    """Scale + crop offsets mapping the ref frame onto the output (cover)."""
    scale = max(out_w / ref_w, out_h / ref_h)
    crop_x = (ref_w * scale - out_w) * anchor[0]
    crop_y = (ref_h * scale - out_h) * anchor[1]
    return scale, crop_x, crop_y


def _raster_projection(out_w, out_h, anchor):
    """Phase-A projection: the calibrated 1400x1050 affine, then cover-crop to output."""
    sc, cx, cy = _cover_transform(geo.REF_W, geo.REF_H, out_w, out_h, anchor)

    def to_px(lon, lat):
        rx, ry = geo.lonlat_to_px(lon, lat)
        return rx * sc - cx, ry * sc - cy

    proj = Projection(
        to_px,
        x_to_lon=lambda x: geo.x_to_lon((x + cx) / sc),
        y_to_lat=lambda y: geo.y_to_lat((y + cy) / sc),
        scale=sc,
    )
    return proj, (sc, cx, cy)


# Vector-map framing. Plain equirectangular over the FULL globe so nothing is cropped —
# the raster's tighter ~333deg window cut off the mid-Pacific (Alaska, Hawaii, the
# Aleutians). Latitude uses the raster art's px-per-degree ratio (|BY|/|AX|), so
# continents keep their familiar (slightly tall) shape rather than the squashed look of a
# 1:1 equirectangular. Centred west of Greenwich so the seam falls in the empty Bering/
# Pacific and the Americas (incl. Alaska) sit comfortably inside the left edge.
VECTOR_LON_CENTER = 12.0
VECTOR_LAT_CENTER = 0.0  # equator-centred → poles at the top/bottom edges (pole-to-pole on a 16:10 panel)


def _vector_projection(out_w, out_h):
    ppd_lon = out_w / 360.0
    ppd_lat = ppd_lon * (abs(geo.BY) / abs(geo.AX))
    cx, cy = out_w / 2.0, out_h / 2.0
    return Projection(
        to_px=lambda lon, lat: (cx + (lon - VECTOR_LON_CENTER) * ppd_lon,
                                cy + (VECTOR_LAT_CENTER - lat) * ppd_lat),
        x_to_lon=lambda x: VECTOR_LON_CENTER + (x - cx) / ppd_lon,
        y_to_lat=lambda y: VECTOR_LAT_CENTER - (y - cy) / ppd_lat,
        scale=out_w / geo.REF_W,
    )


def _arc_half_width(sin_elev, along, across):
    """Half the span of longitude, in degrees, that is darker than `sin_elev` in a row.

    Solar elevation across one row of pixels is

        sin(elev) = along + across * cos(lon - sublon)

    with `along` = sin(lat)sin(dec) and `across` = cos(lat)cos(dec), both fixed for
    that row. Everything darker than the threshold therefore lies in one arc centred
    on the antisolar meridian, and inverting the cosine gives its half width: 180 for
    a row that is entirely dark, 0 for one the shadow never reaches.
    """
    if across <= 1e-12:  # a polar row: elevation does not vary along it
        return 180.0 if along < sin_elev else 0.0
    cos_hour_angle = (sin_elev - along) / across
    if cos_hour_angle >= 1.0:
        return 180.0
    if cos_hour_angle <= -1.0:
        return 0.0
    return 180.0 - math.degrees(math.acos(cos_hour_angle))


def _band_depth(w, h, proj, sublat, sublon, elevations):
    """How many of `elevations` each pixel lies below, as an "L" image.

    `elevations` must run from brightest to darkest; each arc then sits inside the one
    before it, so painting them in order leaves every pixel holding its own depth.

    Drawing the bands as rows of arcs rather than as polygons under a boundary curve
    is what makes this correct everywhere. A curve has to be closed against one edge of
    the canvas, which presumes the dark region reaches a pole — false for the deep
    bands when the sun is within 18 degrees of the equator, and the closure also
    inherited a latent branch error in sun.boundary_lat for a sun south of it.
    """
    dec = math.radians(sublat)
    sin_dec, cos_dec = math.sin(dec), math.cos(dec)
    antisolar_x = proj.to_px(sublon + 180.0, 0.0)[0]
    # Pixels per degree of longitude, and the width of one full turn around the globe.
    # Both projections are affine in longitude, so this is exact. The arcs are drawn at
    # every whole-turn offset as well, since the canvas need not start at the seam.
    px_per_degree = abs(proj.to_px(1.0, 0.0)[0] - proj.to_px(0.0, 0.0)[0])
    period = 360.0 * px_per_degree
    turns = int(w / period) + 2
    copies = [i * period for i in range(-turns, turns + 1)]

    depth = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(depth)

    for y in range(h):
        lat = math.radians(proj.y_to_lat(y + 0.5))
        along = sin_dec * math.sin(lat)
        across = cos_dec * math.cos(lat)
        for k, elev in enumerate(elevations, start=1):
            half = _arc_half_width(math.sin(math.radians(elev)), along, across)
            if half <= 0.0:
                break  # this arc is empty, and every deeper one is narrower still
            half_px = half * px_per_degree
            for shift in copies:
                x0 = max(0.0, antisolar_x - half_px + shift)
                x1 = min(float(w - 1), antisolar_x + half_px + shift)
                if x0 <= x1:
                    draw.rectangle([x0, y, x1, y], fill=k)
    return depth


def _multiply_pow(tint, k):
    """The multiply tint equivalent to `k` stacked multiplies by `tint`."""
    return tuple(round(255.0 * (c / 255.0) ** k) for c in tint)


def _screen_pow(tint, k):
    """The screen tint equivalent to `k` stacked screens by `tint`."""
    return tuple(round(255.0 - 255.0 * (1.0 - c / 255.0) ** k) for c in tint)


# Rows per strip when washing the canvas. Multiply and screen are pixel-wise, so
# strips give the same bytes as one full-canvas blend while its temporaries (the
# tint layer and the blend result) shrink from canvas-sized to strip-sized.
WASH_STRIP_ROWS = 256


def _overlay_night(base, dt, theme, bands, alpha, proj):
    """Composite the day/night terminator with stepped twilight bands.

    Rather than alpha-compositing an opaque dark scrim (a "normal" blend, which mutes
    every pixel toward the same colour and so flattens the map's GMT grid lines and
    country borders), each band is mixed into the base multiplicatively/screen — a colour
    mix that tints toward night / brightens toward day while PRESERVING the contrast of
    fine lines underneath (works for both the raster art and the vector map):
      - day-side LIGHT washes (SCREEN toward the sun) — brighten the lit hemisphere;
      - night-side DARK washes (MULTIPLY toward midnight) — deepen the dark hemisphere.
    The civil/nautical/astronomical elevations make each twilight band a distinct step: a
    pixel `k` bands deep is washed `k` times.

    Multiply and screen each compose to a closed form (_multiply_pow / _screen_pow), so
    one blend per side suffices: take how many bands reach each pixel, turn that count
    into the tint it earns, blend once. Blending band by band instead would allocate a
    full-canvas RGB image per band; the depth map is one byte per pixel, and the blend
    runs strip by strip into `base` in place, so the canvas itself stays the render's
    only full-size RGB allocation.
    """
    w, h = base.size
    sublat, sublon = sun.subsolar_point(dt)
    elevations = TWILIGHT_ELEVATIONS if bands else (0.0,)
    deepest = len(elevations)
    # A pixel below `d` bands is above the other `deepest - d`, so one depth map drives
    # both washes and they cannot disagree about where the terminator runs.
    depth = _band_depth(w, h, proj, sublat, sublon, elevations)

    def wash(lit, tint, op, cumulative):
        # cumulative(tint, 0) is the blend's no-op colour, so untouched pixels pass through.
        steps = [cumulative(tint, k) for k in range(deepest + 1)]
        luts = [[steps[deepest - d if lit else d][c]
                 for d in (min(v, deepest) for v in range(256))]
                for c in range(3)]
        for y0 in range(0, h, WASH_STRIP_ROWS):
            box = (0, y0, w, min(y0 + WASH_STRIP_ROWS, h))
            strip = depth.crop(box)
            layer = Image.merge("RGB", [strip.point(lut) for lut in luts])
            base.paste(op(base.crop(box), layer), box)

    # Day side: SCREEN a light tint (the wash colour scaled by its alpha); black = no-op.
    dw = theme.get("day_wash")
    if dw:
        a = dw[3] if len(dw) > 3 else 255
        tint = tuple(round(c * a / 255) for c in dw[:3])
        wash(lit=True, tint=tint, op=ImageChops.screen, cumulative=_screen_pow)

    # Night side: MULTIPLY toward the night colour; white = no-op. The per-band multiplier
    # is the night colour pulled toward white by `alpha` (so a stack of bands darkens
    # progressively without crushing line contrast the way an opaque scrim would).
    night = theme.get("night")
    if alpha > 0 and night:
        t = alpha / 255.0
        tint = tuple(round(255 - (255 - c) * t) for c in night)
        wash(lit=False, tint=tint, op=ImageChops.multiply, cumulative=_multiply_pow)
    return base


def _recolor_dark(img, light_rgb, thresh=70):
    """Recolour near-black pixels (the wordmark text) to `light_rgb`, keeping coloured
    parts (the IBM bars) intact — so the logo reads on a dark background."""
    px = img.load()
    lr, lg, lb = light_rgb
    for y in range(img.height):
        for x in range(img.width):
            r, g, b, a = px[x, y]
            if a and max(r, g, b) < thresh:
                px[x, y] = (lr, lg, lb, a)
    return img


def _mono_logo(img, rgb):
    """Recolour the whole logo to a single colour (a flat silhouette), keeping its alpha
    (anti-aliased edges preserved). Used for an all-white logo, etc."""
    out = Image.new("RGBA", img.size, tuple(rgb) + (255,))
    out.putalpha(img.getchannel("A"))
    return out


def _draw_logo(canvas, theme, logo_path, bar_height=0, logo_color=None, logo_invert=False,
               logo_scale=1.0, logo_max_height=0.0):
    """Composite the logo, pinned to the bottom-left CORNER of the wallpaper (anchored to
    the canvas, independent of the map framing). Returns its bbox or None.

    `bar_height` lifts it above a status bar overlaying the bottom of the wallpaper.
    `logo_color` (hex) recolours the whole logo to a flat silhouette (e.g. all-white).
    `logo_invert` recolours the near-black pixels to light while keeping other colours —
    handy for a dark wordmark on a dark theme; off by default so colour logos (e.g. Tux)
    composite as-is.
    `logo_max_height` caps the drawn height to that fraction of the canvas height (0 = no
    cap); useful for tall/portrait logos that would otherwise blow up at the fixed width.
    """
    try:
        logo = Image.open(logo_path).convert("RGBA")
    except OSError:
        return None
    # ~10% of canvas width by default; logo_scale lets you size it up/down. A wide
    # wordmark stays short at this width; a square logo (e.g. Tux) reads larger, so
    # logo_scale < 1 is handy there.
    target_w = max(24, round(canvas.width * 0.104 * logo_scale))
    target_h = round(target_w * logo.height / logo.width)
    # Cap the height for tall logos, recomputing width to keep the aspect ratio.
    if logo_max_height and target_h > canvas.height * logo_max_height:
        target_h = round(canvas.height * logo_max_height)
        target_w = max(1, round(target_h * logo.width / logo.height))
    logo = logo.resize((target_w, target_h), Image.LANCZOS)
    mono = _hex(logo_color)
    if mono:
        logo = _mono_logo(logo, mono)
    elif logo_invert:
        logo = _recolor_dark(logo, tuple(theme.get("logo", (235, 235, 235))))
    pad = round(canvas.width * 0.018)
    x, y = pad, canvas.height - target_h - pad - bar_height
    # paste-with-mask, not alpha_composite: over an opaque canvas they compute the
    # same blend, and paste writes in place on RGB instead of allocating a result.
    canvas.paste(logo, (x, y), logo)
    return (x, y, x + target_w, y + target_h)




def _rect_overlap(a, b):
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))


def _place_labels(items, obstacles, bounds, scale):
    """Assign each label a non-overlapping box around its dot (right/left/above/below).

    Greedy: home first, then left-to-right. Each label picks the candidate side with the
    least overlap against obstacles (dots, logo, screen edges) and already-placed labels.
    A city's `label_side` ("left"/"right"/"above"/"below") is tried first; it still falls
    back to another side rather than overlap badly or run off-screen.
    """
    gap = round(6 * scale)
    placed = list(obstacles)
    order = sorted(range(len(items)), key=lambda i: (not items[i]["is_home"], items[i]["px"]))
    default_sides = ["right", "left", "below", "above", "below-right", "below-left"]
    for i in order:
        it = items[i]
        px, py, w, h, g = it["px"], it["py"], it["w"], it["h"], it["dotr"] + gap
        anchors = {
            "right": (px + g, py - h / 2),
            "left": (px - g - w, py - h / 2),
            "below": (px - w / 2, py + g),
            "above": (px - w / 2, py - g - h),
            "below-right": (px + g, py + g),
            "below-left": (px - g - w, py + g),
        }
        pref = it.get("side")
        sides = ([pref] + [s for s in default_sides if s != pref]
                 if pref in anchors else default_sides)
        candidates = [anchors[s] for s in sides]
        best, best_pen = None, None
        for bx, by in candidates:
            box = (bx, by, bx + w, by + h)
            pen = sum(_rect_overlap(box, o) for o in placed)
            off = (max(0, bounds[0] - bx) + max(0, (bx + w) - bounds[2])
                   + max(0, bounds[1] - by) + max(0, (by + h) - bounds[3]))
            pen += off * (w + h) * 3  # heavily penalise going off-screen
            if best_pen is None or pen < best_pen:
                best, best_pen = box, pen
            if pen == 0:
                break
        it["box"] = best
        placed.append(best)


# The vector base map — ocean, land, borders, zone fills, grid, IDL, offset labels —
# depends on nothing that changes between timer ticks, yet building it is nearly all
# of a render's time and memory (it is drawn supersampled, and each tick is a fresh
# process). So it is memoised on disk, keyed by everything build_base reads.
_BASE_THEME_KEYS = ("ocean", "land", "border", "grid", "grid_label", "gmt", "idl", "column")
_BASE_CACHE_KEEP = 4  # e.g. two monitor sizes x a light and a dark theme


def _vector_base(out_w, out_h, theme, font, to_px, home_offset, cache_dir):
    if cache_dir is None:
        return vectormap.build_base(out_w, out_h, theme, font, to_px,
                                    home_offset=home_offset)
    # Code and data identity: the package version plus the drawing module's and the
    # geodata files' paths and mtimes. Under Nix the store path changes on any
    # rebuild; for a pip/editable install the mtimes catch upgrades and edits.
    sources = [vectormap.__file__] + [
        os.path.join(vectormap.GEO_DIR, f) for f in sorted(os.listdir(vectormap.GEO_DIR))
    ]
    # Only stable values may enter the key — never the repr of an arbitrary object.
    # font.path is a filesystem path for fonts loaded by name, but Pillow's bundled
    # fallback (load_default) carries a BytesIO there, whose repr is a fresh memory
    # address per process: hashing it would quietly turn every tick into a miss.
    font_path = getattr(font, "path", None)
    font_id = (font_path if isinstance(font_path, str) else None,
               font.getname() if hasattr(font, "getname") else None,
               getattr(font, "size", None))
    key = hashlib.sha256(repr((
        __version__,
        [(f, os.path.getmtime(f)) for f in sources],
        out_w, out_h,
        {k: tuple(theme[k]) for k in _BASE_THEME_KEYS},
        home_offset,
        font_id,
    )).encode()).hexdigest()[:16]
    path = os.path.join(cache_dir, f"base-{key}.png")

    try:
        img = Image.open(path)
        img.load()
        return img
    except OSError:  # missing, truncated, or unreadable — rebuild and repair below
        pass

    img = vectormap.build_base(out_w, out_h, theme, font, to_px, home_offset=home_offset)
    try:
        _store_base(img, cache_dir, path)
    except OSError as e:
        # The cache is an optimisation: a full disk or unwritable directory must
        # not take the wallpaper down with it. The render itself succeeded.
        print(f"greyline: base-map cache write failed ({e}); continuing uncached",
              file=sys.stderr)
    return img


def _store_base(img, cache_dir, path):
    os.makedirs(cache_dir, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=cache_dir, prefix=".base-", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            img.save(f, format="PNG")
        os.replace(tmp, path)  # atomic: a concurrent reader sees a whole file or none
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    stale = sorted((e for e in os.scandir(cache_dir)
                    if e.name.startswith("base-") and e.name.endswith(".png")),
                   key=lambda e: e.stat().st_mtime, reverse=True)[_BASE_CACHE_KEEP:]
    for e in stale:
        try:
            os.unlink(e.path)
        except OSError:
            pass


def _fmt_time(local, fmt):
    if fmt == "hour":
        # Hour only. On a whole-hour zone the minutes match whatever clock the reader
        # already has, so they add nothing. Zones offset by a fraction of an hour keep
        # theirs — India (+5:30), Nepal (+5:45), Chatham (+12:45) and the rest would
        # otherwise read up to three quarters of an hour wrong.
        if local.utcoffset().total_seconds() % 3600:
            return f"{local.hour:02d}:{local.minute:02d}"
        return f"{local.hour:02d}"
    if fmt == "12h":
        h = local.hour % 12 or 12
        return f"{h}:{local.minute:02d} {'AM' if local.hour < 12 else 'PM'}"
    return f"{local.hour:02d}:{local.minute:02d}"


def _label_lines(city, dt, fmt):
    """City label: name + local time. Kept deliberately simple."""
    local = dt.astimezone(ZoneInfo(city["tz"]))
    return [city["name"], _fmt_time(local, fmt)]


def render(
    cities,
    *,
    dt=None,
    out_size=None,
    theme="modus",
    theme_overrides=None,
    fmt="24h",
    twilight_bands=True,
    darkness="subtle",
    column_highlight=True,
    home_color=None,
    label_bg_alpha=130,
    map_style="vector",
    logo=True,
    logo_path=None,
    logo_color=None,
    logo_invert=False,
    logo_scale=1.0,
    logo_max_height=0.0,
    bar_height=0,
    desaturate=False,
    font_path=None,
    font_bold_path=None,
    font_scale=1.0,
    base_path=BASE_1400,
    crop_anchor=(0.5, 1.0),
    cache_dir=None,  # directory for the vector base-map cache; None renders fresh
):
    th = themes.load_theme(theme, overrides=theme_overrides)
    logo_path = logo_path or LOGO_PNG
    dt = dt or datetime.now(timezone.utc)
    alpha = th.get("night_alpha", DARKNESS_ALPHA.get(darkness, DARKNESS_ALPHA["subtle"]))
    home_rgb = _hex(home_color) or tuple(th["home"])  # accent colour for the home city
    out_w, out_h = out_size or (geo.REF_W, geo.REF_H)

    # Home city + its *standard* (geographic) UTC offset, used to highlight its timezone
    # column. The map's zone polygons are keyed by standard offset (London is always the
    # zone-0 band), so we subtract the DST component out — otherwise the highlight jumps
    # one column east/west for half the year in any DST-observing region (issue #14).
    home = next((c for c in cities if c.get("home")), None)
    home_offset = None
    if home and column_highlight:
        local = dt.astimezone(ZoneInfo(home["tz"]))
        off = local.utcoffset()
        if off is not None:
            std = off - (local.dst() or timedelta(0))
            home_offset = std.total_seconds() / 3600.0

    # Build the map base + a projection (lon/lat -> output px) for the chosen style.
    if map_style == "vector":
        # Full-globe equirectangular (see _vector_projection) — the terminator, column
        # highlight and city clocks all share this one mapping, so they line up.
        proj = _vector_projection(out_w, out_h)
        scale = proj.scale
        grid_font = _load_font(max(8, round(11 * scale)), FONT_CANDIDATES, font_path)
        # The home highlight fills the real zone polygon here (like the GMT column),
        # so the straight-band fallback below is skipped for the vector style.
        canvas = _vector_base(out_w, out_h, th, grid_font, proj.to_px,
                              home_offset, cache_dir)
    else:
        proj, (sc, cx, cy) = _raster_projection(out_w, out_h, crop_anchor)
        scale = sc
        if not os.path.isfile(base_path):
            raise FileNotFoundError(
                f"raster map artwork not found at {base_path}. The IBM/Lenovo 'World Time' "
                "art is not bundled (see NOTICE); use map_style=\"vector\" or supply your own "
                "1400x1050 map via base_path."
            )
        base = Image.open(base_path).convert("RGB")  # the 1400x1050 calibration frame
        if desaturate:  # grayscale the blue artwork → a black-and-white map, then
            # contrast 150% + brightness 70% (darker) for a crisp, muted base.
            gray = ImageOps.grayscale(base)
            gray = ImageEnhance.Contrast(gray).enhance(1.5)
            gray = ImageEnhance.Brightness(gray).enhance(0.7)
            base = gray.convert("RGB")
        scaled = base.resize((round(geo.REF_W * sc), round(geo.REF_H * sc)), Image.LANCZOS)
        canvas = scaled.crop((round(cx), round(cy), round(cx) + out_w, round(cy) + out_h))

    # Home timezone-column highlight for the RASTER style (a straight band, one hour of
    # longitude wide). The vector style fills the real zone polygon in build_base instead.
    if column_highlight and home and map_style != "vector":
        hx, _hy = proj.to_px(home["lon"], home["lat"])
        x0, _ = proj.to_px(home["lon"] - 7.5, home["lat"])
        x1, _ = proj.to_px(home["lon"] + 7.5, home["lat"])
        col_w = abs(x1 - x0)
        col = tuple(th["column"])
        band = Image.new("L", (out_w, out_h), 0)
        ImageDraw.Draw(band).rectangle(
            [hx - col_w / 2, 0, hx + col_w / 2, out_h],
            fill=col[3] if len(col) > 3 else 255,
        )
        ImageDraw.Draw(canvas).bitmap((0, 0), band, fill=col[:3])

    # Day/night + twilight overlay (output space, via the projection).
    canvas = _overlay_night(canvas, dt, th, twilight_bands, alpha, proj)

    # Logo first — its box becomes an obstacle so no label hides behind it.
    obstacles = []
    if logo:
        b = _draw_logo(canvas, th, logo_path, bar_height, logo_color, logo_invert,
                       logo_scale, logo_max_height)
        if b:
            obstacles.append(b)

    # Fonts for the clocks (crisp text, sizes scale with output + font_scale).
    fs = max(8, round(16 * scale * font_scale))
    fs_home = max(10, round(20 * scale * font_scale))
    font = _load_font(fs, FONT_CANDIDATES, font_path)
    font_home = _load_font(fs_home, FONT_BOLD_CANDIDATES, font_bold_path)
    draw = ImageDraw.Draw(canvas)

    # Size each label; collect for placement.
    items = []
    for c in cities:
        px, py = proj.to_px(c["lon"], c["lat"])
        if px < -40 or px > out_w + 40 or py < -40 or py > out_h + 40:
            continue  # off-screen
        is_home = bool(c.get("home"))
        f = font_home if is_home else font
        text = "\n".join(_label_lines(c, dt, fmt))
        bb = draw.multiline_textbbox((0, 0), text, font=f, spacing=2, anchor="la")
        items.append({
            "c": c, "is_home": is_home, "f": f, "text": text, "px": px, "py": py,
            "ox": bb[0], "oy": bb[1], "w": bb[2] - bb[0], "h": bb[3] - bb[1],
            "dotr": round((6 if is_home else 4) * scale),
            "side": c.get("label_side"),  # optional placement preference
        })

    # Place labels avoiding dots, the logo box, the screen edges and each other.
    dot_boxes = [(it["px"] - it["dotr"], it["py"] - it["dotr"],
                  it["px"] + it["dotr"], it["py"] + it["dotr"]) for it in items]
    m = round(10 * scale)
    _place_labels(items, obstacles + dot_boxes,
                  (m, m, out_w - m, out_h - m - bar_height), scale)

    # Semi-transparent rounded backplate behind each label, for legibility over the map.
    # One coverage mask for all plates: plates that overlap must darken once, not twice.
    if label_bg_alpha > 0 and items:
        pad_x = max(4, round(10 * scale * font_scale))
        pad_y = max(3, round(7 * scale * font_scale))
        rad = max(3, round(7 * scale * font_scale))
        plate = Image.new("L", canvas.size, 0)
        pd = ImageDraw.Draw(plate)
        for it in items:
            bx0, by0, bx1, by1 = it["box"]
            pd.rounded_rectangle([bx0 - pad_x, by0 - pad_y, bx1 + pad_x, by1 + pad_y],
                                 radius=rad, fill=label_bg_alpha)
        draw.bitmap((0, 0), plate, fill=(0, 0, 0))

    # Draw dots + labels at their placed boxes.
    for it in items:
        is_home = it["is_home"]
        dot = home_rgb if is_home else th["dot"]
        txt = home_rgb if is_home else th["text"]
        stroke = th["home_stroke"] if is_home else th["text_stroke"]
        r = it["dotr"]
        px, py = it["px"], it["py"]
        draw.ellipse([px - r, py - r, px + r, py + r], fill=dot,
                     outline=th["dot_outline"], width=max(1, round(scale)))
        draw.multiline_text(
            (it["box"][0] - it["ox"], it["box"][1] - it["oy"]), it["text"],
            font=it["f"], fill=txt, spacing=2, anchor="la",
            stroke_width=max(1, round(scale)), stroke_fill=stroke,
        )

    return canvas
