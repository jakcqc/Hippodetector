from __future__ import annotations

try:
    from __main__ import render_press_releases_page
except ImportError:
    from app import render_press_releases_page


render_press_releases_page()
