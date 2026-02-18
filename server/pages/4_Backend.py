from __future__ import annotations

from pathlib import Path
import sys

import streamlit as st

try:
    from __main__ import inject_css
except ImportError:
    try:
        from app import inject_css
    except ModuleNotFoundError:
        sys.path.append(str(Path(__file__).resolve().parents[1]))
        from app import inject_css


BACKEND_IMAGE_PATH = Path(__file__).resolve().parents[1] / "CivicHippo_high_level_1.png"


def main() -> None:
    inject_css()
    st.markdown(
        """
        <div class="hero">
          <p class="hero-title">Backend</p>
          <p class="hero-sub">High-level backend architecture.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not BACKEND_IMAGE_PATH.exists():
        st.error(f"Required image not found: {BACKEND_IMAGE_PATH}")
        return

    _, image_col, _ = st.columns([1, 8, 1])
    with image_col:
        st.image(str(BACKEND_IMAGE_PATH), use_container_width=True)


if __name__ == "__main__":
    main()
