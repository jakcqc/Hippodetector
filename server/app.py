from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import streamlit as st


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
HIPPO_ICON_PATH = Path(__file__).resolve().parent / "HippoD.png"

# Dominant colors extracted from a histogram pass on server/HippoD.png:
# #e090a0, #f0c0b0, #f0b0c0, #c07080, #b06080


def list_json_files() -> list[Path]:
    if not DATA_DIR.exists():
        return []
    return sorted(DATA_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def get_icon_data_url() -> str:
    if not HIPPO_ICON_PATH.exists():
        return ""
    encoded = base64.b64encode(HIPPO_ICON_PATH.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def inject_css() -> None:
    icon_data_url = get_icon_data_url()
    css = """
        <style>
          :root {
            --hippo-bg: #f7f5f8;
            --hippo-surface: #ffffff;
            --hippo-surface-soft: #fff6fa;
            --hippo-border: #f0b0c099;
            --hippo-text: #2f2430;
            --hippo-text-soft: #5d4754;
            --hippo-accent: #c07080;
            --hippo-accent-2: #b06080;
          }
          .stApp {
            background: radial-gradient(1200px 600px at 5% -10%, #f0c0b055 0%, transparent 50%),
                        radial-gradient(900px 500px at 95% -15%, #f0b0c040 0%, transparent 45%),
                        var(--hippo-bg);
            color: var(--hippo-text) !important;
          }
          [data-testid="stHeader"] {
            background: linear-gradient(90deg, #2f2430 0%, #3a2b39 100%) !important;
            min-height: 4.2rem !important;
            border-bottom: 1px solid #b0608055;
          }
          [data-testid="stHeader"]::after {
            content: "";
            position: absolute;
            left: 0.9rem;
            top: 0.45rem;
            width: 3.2rem;
            height: 3.2rem;
            border-radius: 10px;
            background: url("__HIPPO_ICON_URL__") center/cover no-repeat;
            border: 1px solid #f0b0c088;
            box-shadow: 0 4px 12px rgba(0,0,0,0.25);
          }
          [data-testid="stHeader"]::before {
            content: "Where are the Hippos?";
            position: absolute;
            left: 50%;
            transform: translateX(-50%);
            top: 0.65rem;
            font-size: 2rem;
            font-weight: 800;
            letter-spacing: -0.02em;
            color: #f6e9ef;
            white-space: nowrap;
            pointer-events: none;
          }
          [data-testid="stToolbar"] {
            z-index: 5;
          }
          [data-testid="stAppViewContainer"],
          [data-testid="stMain"],
          [data-testid="stMainBlockContainer"] {
            color: var(--hippo-text) !important;
          }
          [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #fff9fc 0%, #fff3f8 100%) !important;
            border-right: 1px solid var(--hippo-border) !important;
          }
          [data-testid="stSidebar"] * {
            color: var(--hippo-text) !important;
          }
          .main .block-container {
            max-width: 1240px;
            padding-top: 2.2rem;
            padding-bottom: 2rem;
          }
          p, span, label, li, div, small {
            color: var(--hippo-text);
          }
          h1, h2, h3, h4, h5, h6 {
            color: var(--hippo-text) !important;
          }
          [data-testid="stCaptionContainer"] p,
          [data-testid="stMarkdownContainer"] p,
          [data-testid="stMarkdownContainer"] li {
            color: var(--hippo-text) !important;
          }
          .hero {
            background: linear-gradient(135deg, #ffffff 0%, #f0c0b022 100%);
            border: 1px solid #f0b0c088;
            border-radius: 16px;
            padding: 1rem 1.2rem;
            margin-bottom: 0.9rem;
            box-shadow: 0 10px 30px rgba(176, 96, 128, 0.16);
          }
          .hero-title {
            font-size: 2rem;
            font-weight: 800;
            letter-spacing: -0.02em;
            color: #2b2028;
            margin: 0;
          }
          .hero-sub {
            color: #5d4754;
            margin-top: 0.35rem;
            margin-bottom: 0;
          }
          div[data-testid="stMetric"] {
            background: var(--hippo-surface);
            border: 1px solid var(--hippo-border);
            border-radius: 14px;
            padding: 0.6rem 0.8rem;
            box-shadow: 0 6px 18px rgba(176, 96, 128, 0.10);
          }
          div[data-testid="stMetricLabel"] {
            color: #5d4754 !important;
          }
          div[data-testid="stMetricLabel"] * {
            color: #5d4754 !important;
            fill: #5d4754 !important;
          }
          div[data-testid="stMetricValue"] {
            color: #2f2430 !important;
          }
          div[data-testid="stMetricValue"] * {
            color: #2f2430 !important;
            fill: #2f2430 !important;
          }
          div[data-testid="stMetricDelta"] {
            color: #6b4f62 !important;
          }
          div[data-testid="stMetricDelta"] * {
            color: #6b4f62 !important;
            fill: #6b4f62 !important;
          }
          div[data-baseweb="select"] * {
            color: var(--hippo-text) !important;
          }
          div[data-baseweb="select"] > div {
            background: var(--hippo-surface) !important;
            border-color: var(--hippo-border) !important;
          }
          div[data-baseweb="select"] input {
            color: var(--hippo-text) !important;
            -webkit-text-fill-color: var(--hippo-text) !important;
          }
          ul[role="listbox"] {
            background: var(--hippo-surface) !important;
            border: 1px solid var(--hippo-border) !important;
          }
          ul[role="listbox"] li {
            color: var(--hippo-text) !important;
            background: var(--hippo-surface) !important;
          }
          ul[role="listbox"] li[aria-selected="true"] {
            background: #f0b0c033 !important;
            color: var(--hippo-text) !important;
          }
          [data-baseweb="input"] > div,
          [data-baseweb="textarea"] > div {
            background: var(--hippo-surface) !important;
            border: 1px solid var(--hippo-border) !important;
          }
          [data-baseweb="input"] input,
          [data-baseweb="textarea"] textarea {
            color: var(--hippo-text) !important;
            -webkit-text-fill-color: var(--hippo-text) !important;
            background: var(--hippo-surface) !important;
          }
          [data-testid="stNumberInput"] label,
          [data-testid="stTextInput"] label,
          [data-testid="stSelectbox"] label {
            color: var(--hippo-text-soft) !important;
          }
          .stTabs [role="tab"] {
            color: #5e4357 !important;
          }
          .stTabs [role="tab"][aria-selected="true"] {
            color: #2f2430 !important;
            font-weight: 600;
          }
          .stTabs [data-baseweb="tab-list"] {
            border-bottom: 1px solid var(--hippo-border);
          }
          [data-testid="stExpander"] {
            background: var(--hippo-surface) !important;
            border: 1px solid var(--hippo-border) !important;
            border-radius: 12px !important;
          }
          [data-testid="stExpander"] summary,
          [data-testid="stExpander"] summary * {
            color: var(--hippo-text) !important;
            background: transparent !important;
          }
          [data-testid="stExpanderDetails"] {
            background: var(--hippo-surface-soft) !important;
            border-top: 1px solid var(--hippo-border);
          }
          .card {
            background: var(--hippo-surface);
            border: 1px solid #f0b0c088;
            border-radius: 14px;
            padding: 0.95rem 1rem;
            box-shadow: 0 6px 18px rgba(176, 96, 128, 0.10);
            color: #2f2430 !important;
          }
          .card * {
            color: #2f2430 !important;
          }
          .card h4 {
            margin-top: 0;
            margin-bottom: 0.65rem;
            color: #c07080;
          }
          .field {
            margin: 0.22rem 0;
            color: #3f3f46;
          }
          .field b {
            color: #b06080;
          }
          .formatted-body {
            background: var(--hippo-surface);
            border: 1px solid #f0b0c088;
            border-radius: 12px;
            padding: 1rem;
            line-height: 1.68;
            font-size: 1rem;
            color: #2f2f35;
          }
          .formatted-body p {
            margin-top: 0;
            margin-bottom: 0.95rem;
          }
          .formatted-body ul, .formatted-body ol {
            margin-top: 0;
            margin-bottom: 0.95rem;
            padding-left: 1.25rem;
          }
          [data-testid="stAlert"] {
            background: var(--hippo-surface) !important;
            color: var(--hippo-text) !important;
            border: 1px solid var(--hippo-border) !important;
          }
          [data-testid="stAlert"] * {
            color: var(--hippo-text) !important;
          }
          [data-testid="stJson"] {
            background: var(--hippo-surface) !important;
            border: 1px solid var(--hippo-border) !important;
            border-radius: 10px !important;
            padding: 0.5rem !important;
          }
          [data-testid="stCodeBlock"] pre {
            background: var(--hippo-surface) !important;
            color: var(--hippo-text) !important;
            border: 1px solid var(--hippo-border) !important;
          }
          a, a:visited {
            color: var(--hippo-accent-2) !important;
          }
        </style>
        """
    st.markdown(css.replace("__HIPPO_ICON_URL__", icon_data_url), unsafe_allow_html=True)


def get_members_map(payload: Any) -> dict[str, dict[str, Any]]:
    if isinstance(payload, dict) and isinstance(payload.get("membersByBioguideId"), dict):
        members_map: dict[str, dict[str, Any]] = {}
        for key, value in payload["membersByBioguideId"].items():
            if isinstance(value, dict):
                members_map[str(key)] = value
        return members_map
    return {}


def flatten_releases(members_map: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    flattened: list[dict[str, Any]] = []
    for bioguide_id, member_data in members_map.items():
        releases = member_data.get("pressReleases")
        if not isinstance(releases, list):
            continue
        for release in releases:
            if isinstance(release, dict):
                flattened.append(
                    {
                        **release,
                        "_bioguideId": bioguide_id,
                        "_memberName": member_data.get("name"),
                        "_memberStatus": member_data.get("status"),
                    }
                )
    return flattened


def preview_row(entry: dict[str, Any], idx: int) -> str:
    title = entry.get("title") or entry.get("url") or f"Entry {idx + 1}"
    date = entry.get("date") or entry.get("publishedTime") or ""
    return f"{idx + 1}. {title}" + (f" ({date})" if date else "")


def member_label(bioguide_id: str, member_data: dict[str, Any]) -> str:
    name = member_data.get("name") or "Unknown"
    status = member_data.get("status") or "-"
    return f"{name} ({bioguide_id}) · {status}"


def render_member_card(member: dict[str, Any], bioguide_id: str) -> None:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h4>Member Profile</h4>", unsafe_allow_html=True)
    st.markdown(f"<div class='field'><b>bioguideId:</b> {bioguide_id}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='field'><b>name:</b> {member.get('name', '-')}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='field'><b>lastName:</b> {member.get('lastName', '-')}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='field'><b>state:</b> {member.get('state', '-')}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='field'><b>partyName:</b> {member.get('partyName', '-')}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='field'><b>status:</b> {member.get('status', '-')}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='field'><b>source:</b> {member.get('source', '-')}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='field'><b>pagesScraped:</b> {member.get('pagesScraped', '-')}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='field'><b>releaseCount:</b> {member.get('releaseCount', '-')}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def main() -> None:
    page_icon: str | Path = "🦛"
    if HIPPO_ICON_PATH.exists():
        page_icon = HIPPO_ICON_PATH
    st.set_page_config(page_title="Hippodetector Press Release Viewer", page_icon=page_icon, layout="wide")
    inject_css()

    st.markdown(
        """
        <div class="hero">
          <p class="hero-title">Where are the Hippos?</p>
          <p class="hero-sub">Soft view for browsing Congress member releases keyed by bioguideId.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    files = list_json_files()
    if not files:
        st.error(f"No JSON files found in {DATA_DIR}")
        return

    with st.sidebar:
        st.header("Data Source")
        selected_file = st.selectbox("JSON file", [p.name for p in files], index=0)

    selected_path = next(p for p in files if p.name == selected_file)
    payload = load_json(selected_path)
    members_map = get_members_map(payload)

    if not members_map:
        st.warning("This file is not in `membersByBioguideId` format.")
        st.json(payload)
        return

    sorted_members = sorted(members_map.items(), key=lambda kv: str(kv[1].get("name") or ""))
    labels = ["All members"] + [member_label(bg, md) for bg, md in sorted_members]
    label_to_id = {"All members": "ALL"}
    for bg, md in sorted_members:
        label_to_id[member_label(bg, md)] = bg

    with st.sidebar:
        selected_member_label = st.selectbox("Member", labels, index=0)
        query = st.text_input("Filter release text/title", "")
        max_releases_to_show = st.number_input(
            "Max releases shown",
            min_value=1,
            max_value=10000,
            value=200,
            step=25,
        )

    selected_member_id = label_to_id[selected_member_label]

    all_releases = flatten_releases(members_map)
    filtered_releases = all_releases
    if selected_member_id != "ALL":
        filtered_releases = [r for r in filtered_releases if r.get("_bioguideId") == selected_member_id]

    if query:
        q = query.lower()
        filtered_releases = [
            r for r in filtered_releases if q in json.dumps(r, ensure_ascii=False).lower()
        ]
    filtered_releases = filtered_releases[: int(max_releases_to_show)]

    counts = payload.get("statusCounts", {}) if isinstance(payload, dict) else {}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Members", len(members_map))
    c2.metric("Releases Shown", len(filtered_releases))
    c3.metric("OK Members", str(counts.get("ok", "-")))
    c4.metric("Error Members", str(counts.get("error", "-")))

    if not filtered_releases:
        st.info("No releases match current filters.")
        return

    if selected_member_id == "ALL":
        st.caption(
            f"Showing up to {int(max_releases_to_show)} releases across all members. "
            "Select a specific member to scope this limit per member."
        )
    else:
        member_name = members_map[selected_member_id].get("name", selected_member_id)
        st.caption(
            f"Showing {len(filtered_releases)} releases for {member_name} "
            f"(max {int(max_releases_to_show)})."
        )

    left, right = st.columns([1.15, 2.85])

    with left:
        if selected_member_id != "ALL":
            render_member_card(members_map[selected_member_id], selected_member_id)
        else:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<h4>Member Profile</h4>", unsafe_allow_html=True)
            st.markdown(
                "<div class='field'>Select a specific member to view full profile fields "
                "(bioguideId, name, lastName, state, partyName, status, source).</div>",
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.subheader("Releases")
        for idx, release in enumerate(filtered_releases):
            title = release.get("title") or "Untitled release"
            date = release.get("date") or release.get("publishedTime") or "-"
            bg = release.get("_bioguideId", "-")
            member_name = release.get("_memberName", "-")
            header = f"{idx + 1}. {title} ({date})"
            with st.expander(header, expanded=(idx == 0)):
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown(f"<div class='field'><b>Member:</b> {member_name} ({bg})</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='field'><b>URL:</b> {release.get('url', '-')}</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

                tab_text, tab_raw = st.tabs(["Text", "Raw JSON"])
                with tab_text:
                    body_html = release.get("bodyHtml")
                    body_text = release.get("bodyText")
                    if body_html:
                        st.markdown(f"<div class='formatted-body'>{body_html}</div>", unsafe_allow_html=True)
                    elif body_text:
                        st.text_area(f"bodyText_{idx}", body_text, height=420)
                    else:
                        st.write("No body content available.")
                with tab_raw:
                    st.json(release)


if __name__ == "__main__":
    main()
