from __future__ import annotations

import json
import html
from pathlib import Path
import sys
from typing import Any

from rapidfuzz import fuzz, process
import streamlit as st

try:
    from app import HIPPO_ICON_PATH, inject_css, normalize_text
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from app import HIPPO_ICON_PATH, inject_css, normalize_text


DATA_DIR = Path(__file__).resolve().parents[2] / "data"
MEMBERS_DIR = DATA_DIR / "members"
TOPIC_ORDER = [
    "abortion",
    "budget_economy",
    "civil_rights",
    "corporations",
    "crime",
    "drugs",
    "education",
    "energy_oil",
    "environment",
    "families_children",
    "foreign_policy",
    "free_trade",
    "government_reform",
    "gun_control",
    "health_care",
    "homeland_security",
    "immigration",
    "jobs",
    "principles_values",
    "social_security",
    "tax_reform",
    "technology",
    "war_peace",
    "welfare_poverty",
]
STATUS_CODE = {
    "supports": "S",
    "opposes": "O",
    "mixed": "M",
    "unknown": "U",
    "no_stance_recorded": "-",
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def stance_search_text(stance_record: dict[str, Any]) -> str:
    press_release = stance_record.get("press_release") if isinstance(stance_record.get("press_release"), dict) else {}
    stance_map = stance_record.get("stances") if isinstance(stance_record.get("stances"), dict) else {}
    text_parts = [
        str(stance_record.get("_memberName") or ""),
        str(stance_record.get("_bioguideId") or ""),
        str(press_release.get("title") or ""),
        str(press_release.get("date") or ""),
        str(press_release.get("url") or ""),
    ]
    for issue, stance in stance_map.items():
        text_parts.append(str(issue))
        if isinstance(stance, dict):
            text_parts.append(str(stance.get("issue") or ""))
            text_parts.append(str(stance.get("status") or ""))
            text_parts.append(str(stance.get("stance") or "")[:900])
            text_parts.append(str(stance.get("summary") or "")[:900])
    return normalize_text(" ".join(text_parts))


@st.cache_data(show_spinner=False)
def build_stance_index(members_dir_path: str, signature: str) -> dict[str, Any]:
    del signature
    members_dir = Path(members_dir_path)
    if not members_dir.exists():
        return {"stances": []}

    all_stances: list[dict[str, Any]] = []
    for stances_file in sorted(members_dir.glob("*_stances.json")):
        try:
            payload = load_json(stances_file)
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
        bioguide_id = str(metadata.get("bioguideId") or stances_file.stem.replace("_stances", ""))
        member_name = str(metadata.get("name") or bioguide_id)
        member_state = str(metadata.get("state") or "-")
        member_party = str(metadata.get("party") or "-")
        stances = payload.get("stances")
        if isinstance(stances, list):
            for record in stances:
                if not isinstance(record, dict):
                    continue
                enriched = {
                    **record,
                    "_bioguideId": bioguide_id,
                    "_memberName": member_name,
                    "_memberState": member_state,
                    "_memberParty": member_party,
                }
                enriched["_search_text"] = stance_search_text(enriched)
                all_stances.append(enriched)
            continue

        # New format: a single stances object keyed by topic.
        if isinstance(stances, dict):
            normalized_stances: dict[str, dict[str, str]] = {}
            for topic, detail in stances.items():
                if not isinstance(detail, dict):
                    continue
                normalized_stances[str(topic)] = {
                    "issue": str(detail.get("issue") or ""),
                    "stance": str(detail.get("stance") or ""),
                    "status": "unknown",
                    "summary": str(detail.get("stance") or ""),
                    "source_url": "",
                }
            enriched = {
                "press_release": {},
                "stances": normalized_stances,
                "_bioguideId": bioguide_id,
                "_memberName": member_name,
                "_memberState": member_state,
                "_memberParty": member_party,
            }
            enriched["_search_text"] = stance_search_text(enriched)
            all_stances.append(enriched)
    return {"stances": all_stances}


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


def get_topic_status(record: dict[str, Any], topic: str) -> str:
    stance_map = record.get("stances") if isinstance(record.get("stances"), dict) else {}
    stance = stance_map.get(topic) if isinstance(stance_map.get(topic), dict) else {}
    raw = str(stance.get("status") or "no_stance_recorded").strip().lower()
    return STATUS_CODE.get(raw, "U")


def topic_display_name(topic: str) -> str:
    return topic.replace("_", " ").title()


def get_topic_detail_rows(record: dict[str, Any], fallback_url: str) -> list[dict[str, str]]:
    del fallback_url
    stance_map = record.get("stances") if isinstance(record.get("stances"), dict) else {}
    rows: list[dict[str, str]] = []
    for topic in TOPIC_ORDER:
        stance = stance_map.get(topic) if isinstance(stance_map.get(topic), dict) else {}
        status = str(stance.get("status") or "no_stance_recorded")
        summary = str(stance.get("summary") or stance.get("stance") or "")
        rows.append(
            {
                "topic": topic,
                "issue": str(stance.get("issue") or ""),
                "status": status,
                "summary": summary,
            }
        )
    return rows


def main() -> None:
    page_icon: str | Path = "🦛"
    if HIPPO_ICON_PATH.exists():
        page_icon = HIPPO_ICON_PATH
    st.set_page_config(page_title="Hippodetector Stances", page_icon=page_icon, layout="wide")
    inject_css()

    st.markdown(
        """
        <div class="hero">
          <p class="hero-title">Stances</p>
          <p class="hero-sub">Browse extracted issue stances from press releases.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    stance_files = sorted(MEMBERS_DIR.glob("*_stances.json")) if MEMBERS_DIR.exists() else []
    if not stance_files:
        st.error(f"No stance files found in {MEMBERS_DIR} matching *_stances.json")
        return

    stance_signature = "|".join(f"{p.name}:{p.stat().st_mtime_ns}" for p in stance_files)
    stance_index = build_stance_index(str(MEMBERS_DIR), stance_signature)
    all_stances = stance_index["stances"]
    if not all_stances:
        st.info("No stance records found.")
        return

    members = sorted({str(s.get("_memberName") or "-") for s in all_stances})
    parties = sorted({str(s.get("_memberParty") or "-") for s in all_stances})
    states = sorted({str(s.get("_memberState") or "-") for s in all_stances})

    st.subheader("Navigator")
    nav1, nav2, nav3, nav4 = st.columns([2.1, 1.3, 1.0, 1.0])
    with nav1:
        query = st.text_input("Fuzzy search", "", placeholder="Try: government_reform supports")
    with nav2:
        selected_member = st.selectbox("Member", ["All members"] + members, index=0)
    with nav3:
        selected_party = st.selectbox("Party", ["All parties"] + parties, index=0)
    with nav4:
        selected_state = st.selectbox("State", ["All states"] + states, index=0)

    base_indices = list(range(len(all_stances)))
    filtered_indices = []
    for idx in base_indices:
        item = all_stances[idx]
        member_ok = selected_member == "All members" or str(item.get("_memberName") or "-") == selected_member
        party_ok = selected_party == "All parties" or str(item.get("_memberParty") or "-") == selected_party
        state_ok = selected_state == "All states" or str(item.get("_memberState") or "-") == selected_state
        if member_ok and party_ok and state_ok:
            filtered_indices.append(idx)

    search_texts = [str(item.get("_search_text") or "") for item in all_stances]
    search_signature = (stance_signature, selected_member, selected_party, selected_state, normalize_text(query))
    if st.session_state.get("stances_search_signature") != search_signature:
        st.session_state["stances_filtered_indices"] = fuzzy_filter_indices(filtered_indices, search_texts, query, 0.42)
        st.session_state["stances_search_signature"] = search_signature
        st.session_state["stances_page"] = 1
    matched_indices = st.session_state.get("stances_filtered_indices", filtered_indices)

    topic_groups: dict[str, list[dict[str, str]]] = {topic: [] for topic in TOPIC_ORDER}
    for stance_idx in matched_indices:
        record = all_stances[stance_idx]
        detail_rows = get_topic_detail_rows(record, "")
        for row in detail_rows:
            topic = str(row.get("topic") or "")
            status_raw = str(row.get("status") or "no_stance_recorded").strip().lower()
            status_code = STATUS_CODE.get(status_raw, "U")
            summary = str(row.get("summary") or "").strip()
            if status_code == "-" and not summary:
                continue
            topic_groups[topic].append(
                {
                    "status": status_raw,
                    "status_code": status_code,
                    "summary": summary or "No summary extracted.",
                }
            )

    filtered_topics = [(topic, rows) for topic, rows in topic_groups.items() if rows]
    total_results = sum(len(rows) for _, rows in filtered_topics)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Stance files", len(stance_files))
    c2.metric("Stance records", len(all_stances))
    c3.metric("Matched records", len(matched_indices))
    c4.metric("Matched topic stances", total_results)

    if total_results == 0:
        st.info("No topic stances match current filters.")
        return

    st.caption("Status codes: S=supports, O=opposes, M=mixed, U=unknown, -=no_stance_recorded")
    topic_labels = [
        f"{topic_display_name(topic)} ({len(rows)})"
        for topic, rows in filtered_topics
    ]
    topic_tabs = st.tabs(topic_labels)

    for i, (topic, rows) in enumerate(filtered_topics):
        with topic_tabs[i]:
            combined_summary = " ".join(str(row.get("summary") or "").strip() for row in rows if str(row.get("summary") or "").strip())
            if combined_summary:
                st.markdown("**Summaries**")
                st.markdown(f"<div class='formatted-body'>{html.escape(combined_summary)}</div>", unsafe_allow_html=True)
            else:
                st.info("No summaries available for this topic.")


if __name__ == "__main__":
    main()
