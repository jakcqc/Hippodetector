from __future__ import annotations

import base64
import json
import math
from pathlib import Path
import random
import re
from typing import Any

from rapidfuzz import fuzz, process
import streamlit as st


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
PRESS_RELEASES_FILE = DATA_DIR / "press_releases_by_bioguide.json"
HIPPO_ICON_PATH = Path(__file__).resolve().parent / "HippoD.png"

# Dominant colors extracted from a histogram pass on server/HippoD.png:
# #e090a0, #f0c0b0, #f0b0c0, #c07080, #b06080

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
            --hippo-control-bg: #2b2230;
            --hippo-control-bg-hover: #3a2e41;
            --hippo-control-border: #604a69;
            --hippo-control-text: #ffffff;
            --hippo-control-text-soft: #f4eaf0;
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
          div[data-baseweb="select"] > div {
            background: var(--hippo-control-bg) !important;
            border-color: var(--hippo-control-border) !important;
            color: var(--hippo-control-text) !important;
          }
          div[data-baseweb="popover"],
          div[data-baseweb="menu"] {
            background: var(--hippo-control-bg) !important;
            border: 1px solid var(--hippo-control-border) !important;
            color: var(--hippo-control-text) !important;
          }
          div[data-baseweb="popover"] *,
          div[data-baseweb="menu"] * {
            color: var(--hippo-control-text) !important;
            fill: var(--hippo-control-text) !important;
          }
          div[data-baseweb="select"] [role="combobox"],
          div[data-baseweb="select"] span,
          div[data-baseweb="select"] svg {
            color: var(--hippo-control-text) !important;
            fill: var(--hippo-control-text) !important;
            -webkit-text-fill-color: var(--hippo-control-text) !important;
          }
          div[data-baseweb="select"] input {
            color: var(--hippo-control-text) !important;
            -webkit-text-fill-color: var(--hippo-control-text) !important;
          }
          ul[role="listbox"] {
            background: var(--hippo-control-bg) !important;
            border: 1px solid var(--hippo-control-border) !important;
          }
          ul[role="listbox"] li {
            color: var(--hippo-control-text) !important;
            background: var(--hippo-control-bg) !important;
          }
          ul[role="listbox"] li[aria-selected="true"] {
            background: var(--hippo-control-bg-hover) !important;
            color: var(--hippo-control-text) !important;
          }
          div[role="option"] {
            background: var(--hippo-control-bg) !important;
            color: var(--hippo-control-text) !important;
          }
          div[role="option"][aria-selected="true"] {
            background: var(--hippo-control-bg-hover) !important;
            color: var(--hippo-control-text) !important;
          }
          [data-testid="stSelectbox"] * {
            color: var(--hippo-control-text) !important;
          }
          [data-testid="stSelectbox"] label {
            color: var(--hippo-control-text-soft) !important;
          }
          div[data-baseweb="select"] > div:hover,
          div[data-baseweb="select"] > div:focus-within {
            border-color: #8f6c9b !important;
            box-shadow: 0 0 0 1px #8f6c9b55 !important;
          }
          .stButton > button,
          [data-testid="stButton"] > button,
          button[data-testid^="baseButton-"] {
            background: var(--hippo-control-bg) !important;
            border: 1px solid var(--hippo-control-border) !important;
            color: var(--hippo-control-text) !important;
            -webkit-text-fill-color: var(--hippo-control-text) !important;
          }
          .stButton > button:hover,
          [data-testid="stButton"] > button:hover,
          button[data-testid^="baseButton-"]:hover {
            background: var(--hippo-control-bg-hover) !important;
            border-color: #8f6c9b !important;
            color: var(--hippo-control-text) !important;
            -webkit-text-fill-color: var(--hippo-control-text) !important;
          }
          .stButton > button:focus,
          [data-testid="stButton"] > button:focus,
          button[data-testid^="baseButton-"]:focus {
            color: var(--hippo-control-text) !important;
            -webkit-text-fill-color: var(--hippo-control-text) !important;
            box-shadow: 0 0 0 1px #8f6c9b66 !important;
          }
          .stButton > button:disabled,
          .stButton > button[disabled],
          [data-testid="stButton"] > button:disabled,
          [data-testid="stButton"] > button[disabled],
          button[data-testid^="baseButton-"]:disabled,
          button[data-testid^="baseButton-"][disabled] {
            background: #3a3440 !important;
            border-color: #5f5567 !important;
            color: #e8dfe6 !important;
            -webkit-text-fill-color: #e8dfe6 !important;
            opacity: 0.95 !important;
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
          [data-testid="stTextInput"] label {
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


@st.cache_data(show_spinner=False)
def build_release_index(data_path: str, mtime_ns: int) -> dict[str, Any]:
    del mtime_ns
    payload = load_json(Path(data_path))
    members_map = get_members_map(payload)
    selectable_members = [
        (bg, md)
        for bg, md in members_map.items()
        if str(md.get("status") or "").strip().lower() != "error"
    ]

    releases: list[dict[str, Any]] = []
    search_texts: list[str] = []
    for bioguide_id, member_data in selectable_members:
        member_releases = member_data.get("pressReleases")
        if not isinstance(member_releases, list):
            continue
        for release in member_releases:
            if not isinstance(release, dict):
                continue
            enriched = {
                **release,
                "_bioguideId": bioguide_id,
                "_memberName": member_data.get("name"),
                "_memberStatus": member_data.get("status"),
            }
            search_text = release_search_text(enriched)
            enriched["_search_text"] = search_text
            releases.append(enriched)
            search_texts.append(search_text)

    return {
        "payload": payload,
        "members_map": members_map,
        "selectable_members": selectable_members,
        "releases": releases,
        "search_texts": search_texts,
    }


def member_label(bioguide_id: str, member_data: dict[str, Any]) -> str:
    name = member_data.get("name") or "Unknown"
    status = member_data.get("status") or "-"
    return f"{name} ({bioguide_id}) · {status}"


def normalize_text(value: str) -> str:
    return " ".join(value.lower().split())


def strip_html(value: str) -> str:
    return re.sub(r"<[^>]+>", " ", value)


def release_search_text(release: dict[str, Any]) -> str:
    title = str(release.get("title") or "")
    member_name = str(release.get("_memberName") or "")
    bioguide_id = str(release.get("_bioguideId") or "")
    date = str(release.get("date") or release.get("publishedTime") or "")
    url = str(release.get("url") or "")
    body_text = str(release.get("bodyText") or "")[:800]
    body_html = strip_html(str(release.get("bodyHtml") or ""))[:800]
    return normalize_text(" ".join([title, member_name, bioguide_id, date, url, body_text, body_html]))


def fuzzy_filter_indices(
    search_base_indices: list[int],
    search_texts: list[str],
    query: str,
    threshold: float = 0.42,
) -> list[int]:
    if not query.strip():
        return search_base_indices

    normalized_query = normalize_text(query)
    if not normalized_query:
        return search_base_indices

    choices = [search_texts[idx] for idx in search_base_indices]
    matches = process.extract(
        normalized_query,
        choices,
        scorer=fuzz.token_set_ratio,
        score_cutoff=int(threshold * 100),
        limit=None,
    )
    return [search_base_indices[match_idx] for _, _, match_idx in matches]


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


def render_selection_context_card(
    *,
    selected_file: str,
    selected_party: str,
    selected_state: str,
    selected_member_id: str,
    filtered_member_count: int,
    total_selectable_members: int,
    search_scope: str,
) -> None:
    selected_member = "All members" if selected_member_id == "ALL" else selected_member_id
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h4>Selection Context</h4>", unsafe_allow_html=True)
    st.markdown(f"<div class='field'><b>data file:</b> {selected_file}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='field'><b>party filter:</b> {selected_party}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='field'><b>state filter:</b> {selected_state}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='field'><b>member selection:</b> {selected_member}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='field'><b>search scope:</b> {search_scope}</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='field'><b>members in context:</b> {filtered_member_count} / {total_selectable_members}</div>",
        unsafe_allow_html=True,
    )
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

    if not PRESS_RELEASES_FILE.exists():
        st.error(f"Required file not found: {PRESS_RELEASES_FILE}")
        return

    dataset_mtime_ns = PRESS_RELEASES_FILE.stat().st_mtime_ns
    index_data = build_release_index(str(PRESS_RELEASES_FILE), dataset_mtime_ns)
    payload = index_data["payload"]
    members_map = index_data["members_map"]
    selectable_members = index_data["selectable_members"]
    all_releases = index_data["releases"]
    search_texts = index_data["search_texts"]

    if not members_map:
        st.warning("`press_releases_by_bioguide.json` is not in `membersByBioguideId` format.")
        return
    all_parties = sorted({str(md.get("partyName") or "-") for _, md in selectable_members})
    all_states = sorted({str(md.get("state") or "-") for _, md in selectable_members})

    st.subheader("Navigator")
    nav1, nav2, nav3, nav4, nav5, nav6 = st.columns([2.1, 1.2, 1.0, 1.5, 1.2, 0.9])
    with nav2:
        selected_party = st.selectbox("Party", ["All parties"] + all_parties, index=0)
    with nav3:
        selected_state = st.selectbox("State", ["All states"] + all_states, index=0)
    with nav4:
        search_scope = st.selectbox(
            "Search scope",
            ["Current selection", "All members"],
            index=0,
            help="Use 'All members' to search all members in the current Party/State context.",
        )
    with nav5:
        query = st.text_input("Fuzzy search", "", placeholder="Try: border security")
    with nav6:
        page_size = st.number_input("Releases/page", min_value=1, max_value=500, value=25, step=5)

    filtered_members = []
    for bg, md in selectable_members:
        party_name = str(md.get("partyName") or "-")
        state_name = str(md.get("state") or "-")
        party_ok = selected_party == "All parties" or party_name == selected_party
        state_ok = selected_state == "All states" or state_name == selected_state
        if party_ok and state_ok:
            filtered_members.append((bg, md))

    filtered_members_map = {bg: md for bg, md in filtered_members}
    sorted_members = sorted(filtered_members, key=lambda kv: str(kv[1].get("name") or ""))
    labels = ["All members"] + [member_label(bg, md) for bg, md in sorted_members]
    label_to_id = {"All members": "ALL"}
    id_to_label = {"ALL": "All members"}
    for bg, md in sorted_members:
        label = member_label(bg, md)
        label_to_id[label] = bg
        id_to_label[bg] = label

    random_member_key = "random_default_member_id"
    random_member_signature_key = "random_default_member_signature"
    selected_member_state_key = "selected_member_id"
    dataset_signature = f"{PRESS_RELEASES_FILE}:{dataset_mtime_ns}"
    if st.session_state.get(random_member_signature_key) != dataset_signature:
        st.session_state[random_member_signature_key] = dataset_signature
        st.session_state[random_member_key] = random.choice([bg for bg, _ in selectable_members]) if selectable_members else "ALL"
        st.session_state[selected_member_state_key] = st.session_state[random_member_key]

    if filtered_members and st.session_state.get(selected_member_state_key) not in filtered_members_map:
        st.session_state[selected_member_state_key] = random.choice([bg for bg, _ in filtered_members])
    if not filtered_members:
        st.session_state[selected_member_state_key] = "ALL"

    with nav1:
        if len(labels) == 1:
            st.warning("No members match Party/State filters.")
            selected_member_label = "All members"
        else:
            default_member_id = st.session_state.get(selected_member_state_key, "ALL")
            default_label = id_to_label.get(default_member_id, "All members")
            selected_member_label = st.selectbox(
                "Search/select member",
                labels,
                index=labels.index(default_label),
                help="Type in this box to autocomplete by member name or bioguideId.",
            )

    selected_member_id = label_to_id[selected_member_label]
    st.session_state[selected_member_state_key] = selected_member_id

    filtered_member_ids = set(filtered_members_map.keys())
    context_release_indices = [
        idx for idx, release in enumerate(all_releases) if str(release.get("_bioguideId")) in filtered_member_ids
    ]
    selected_member_release_indices = context_release_indices
    if selected_member_id != "ALL":
        selected_member_release_indices = [
            idx for idx in context_release_indices if str(all_releases[idx].get("_bioguideId")) == selected_member_id
        ]

    search_base_indices = selected_member_release_indices
    if search_scope == "All members":
        search_base_indices = context_release_indices

    search_signature = (
        dataset_signature,
        selected_member_id,
        selected_party,
        selected_state,
        search_scope,
        normalize_text(query),
    )
    if st.session_state.get("release_search_signature") != search_signature:
        st.session_state["filtered_release_indices"] = fuzzy_filter_indices(
            search_base_indices,
            search_texts,
            query,
            0.42,
        )
        st.session_state["release_search_signature"] = search_signature
        st.session_state["release_page"] = 1
    filtered_release_indices = st.session_state.get("filtered_release_indices", search_base_indices)

    total_results = len(filtered_release_indices)
    total_pages = max(1, math.ceil(total_results / int(page_size)))
    current_page = int(st.session_state.get("release_page", 1))
    current_page = max(1, min(current_page, total_pages))
    st.session_state["release_page"] = current_page

    pager1, pager2, pager3, pager4 = st.columns([1, 1.3, 1, 1.2])
    with pager1:
        prev_clicked = st.button("Prev page", disabled=current_page <= 1)
    with pager2:
        page_input = int(
            st.number_input("Page", min_value=1, max_value=total_pages, value=current_page, step=1)
        )
    with pager3:
        next_clicked = st.button("Next page", disabled=current_page >= total_pages)
    with pager4:
        st.caption(f"Page {current_page} of {total_pages}")

    if prev_clicked and current_page > 1:
        current_page -= 1
    if next_clicked and current_page < total_pages:
        current_page += 1
    if not prev_clicked and not next_clicked:
        current_page = page_input
    current_page = max(1, min(current_page, total_pages))
    st.session_state["release_page"] = current_page

    start_idx = (current_page - 1) * int(page_size)
    end_idx = start_idx + int(page_size)
    paged_release_indices = filtered_release_indices[start_idx:end_idx]
    paged_releases = [all_releases[idx] for idx in paged_release_indices]

    counts = payload.get("statusCounts", {}) if isinstance(payload, dict) else {}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Members", len(members_map))
    c2.metric("Members In Context", len(filtered_members))
    c3.metric("Releases Matched", total_results)
    c4.metric("OK Members", str(counts.get("ok", "-")))

    if not paged_releases:
        st.info("No releases match current filters.")
        return

    if search_scope == "All members":
        st.caption(
            f"Showing releases {start_idx + 1}-{min(end_idx, total_results)} of {total_results} "
            "from all members in the current Party/State context (fuzzy search enabled)."
        )
    elif selected_member_id == "ALL":
        st.caption(
            f"Showing releases {start_idx + 1}-{min(end_idx, total_results)} of {total_results} "
            "across all members in the current selection."
        )
    else:
        member_name = members_map[selected_member_id].get("name", selected_member_id)
        st.caption(
            f"Showing releases {start_idx + 1}-{min(end_idx, total_results)} of {total_results} "
            f"for {member_name} (search covers non-visible pages too)."
        )

    left, right = st.columns([1.15, 2.85])

    with left:
        render_selection_context_card(
            selected_file=PRESS_RELEASES_FILE.name,
            selected_party=selected_party,
            selected_state=selected_state,
            selected_member_id=selected_member_id,
            filtered_member_count=len(filtered_members),
            total_selectable_members=len(selectable_members),
            search_scope=search_scope,
        )
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
        for local_idx, release in enumerate(paged_releases, start=0):
            global_idx = start_idx + local_idx
            release_idx = paged_release_indices[local_idx]
            title = release.get("title") or "Untitled release"
            date = release.get("date") or release.get("publishedTime") or "-"
            bg = release.get("_bioguideId", "-")
            member_name = release.get("_memberName", "-")
            header = f"{global_idx + 1}. {title} ({date})"
            with st.expander(header, expanded=(local_idx == 0)):
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown(f"<div class='field'><b>Member:</b> {member_name} ({bg})</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='field'><b>URL:</b> {release.get('url', '-')}</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
                body_html = release.get("bodyHtml")
                body_text = release.get("bodyText")
                preview_source = body_text or strip_html(str(body_html or ""))
                preview_text = " ".join(str(preview_source).split())[:280]
                if preview_text:
                    st.caption(preview_text + ("..." if len(str(preview_source)) > 280 else ""))
                if body_html:
                    st.markdown(f"<div class='formatted-body'>{body_html}</div>", unsafe_allow_html=True)
                elif body_text:
                    st.markdown(f"<div class='formatted-body'>{body_text}</div>", unsafe_allow_html=True)
                else:
                    st.write("No body content available.")


if __name__ == "__main__":
    main()
