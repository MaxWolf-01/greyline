#!/usr/bin/env python3
"""Peak-RSS and wall-time for one render, at the resolution that hurts.

    python tests/bench_render.py [WIDTHxHEIGHT]

Reports tracemalloc's peak Python-side allocation and the process's peak RSS
(ru_maxrss). Pillow buffers land in RSS but not in tracemalloc, so RSS is the
number that matters here.
"""
import resource
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, ".")
from worldtime import render  # noqa: E402

CITIES = [
    {"name": "Vienna", "lat": 48.21, "lon": 16.37, "tz": "Europe/Vienna", "home": True},
    {"name": "San Francisco", "lat": 37.77, "lon": -122.42, "tz": "America/Los_Angeles"},
    {"name": "New York", "lat": 40.71, "lon": -74.01, "tz": "America/New_York"},
    {"name": "London", "lat": 51.51, "lon": -0.13, "tz": "Europe/London"},
    {"name": "Moscow", "lat": 55.76, "lon": 37.62, "tz": "Europe/Moscow"},
    {"name": "Tokyo", "lat": 35.68, "lon": 139.69, "tz": "Asia/Tokyo"},
    {"name": "Sydney", "lat": -33.87, "lon": 151.21, "tz": "Australia/Sydney"},
]


def main():
    size = sys.argv[1] if len(sys.argv) > 1 else "3840x2400"
    w, h = (int(v) for v in size.lower().split("x"))
    dt = datetime(2026, 8, 16, 9, 0, tzinfo=timezone.utc)

    t0 = time.perf_counter()
    img = render.render(CITIES, dt=dt, out_size=(w, h), map_style="vector", logo=False)
    elapsed = time.perf_counter() - t0
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

    print(f"{size}  {elapsed:6.2f}s  peak RSS {peak_rss:7.1f} MB  out {img.size}")


if __name__ == "__main__":
    main()
