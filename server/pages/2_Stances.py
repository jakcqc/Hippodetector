from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from typing import Any

from dotenv import load_dotenv
import httpx
from rapidfuzz import fuzz, process
import streamlit as st

try:
    from __main__ import get_preloaded_json, inject_css, normalize_text
except ImportError:
    try:
        from app import get_preloaded_json, inject_css, normalize_text
    except ModuleNotFoundError:
        sys.path.append(str(Path(__file__).resolve().parents[1]))
        from app import get_preloaded_json, inject_css, normalize_text


DATA_DIR = Path(__file__).resolve().parents[2] / "data"
STANCES_DIR = DATA_DIR / "stances"
PRESS_RELEASES_FILE = DATA_DIR / "press_releases_by_bioguide.json"
SOURCE_PUBLIC_RELEASE = "public_release"
SOURCE_VOTING_RECORD = "voting_record"
PUBLIC_ISSUE_FILE_SUFFIXES = ("_issue_profile.json", "_issues_profile.json")
VOTING_ISSUE_FILE_SUFFIXES = ("_voting_issue_profile.json", "_voting_issues_profile.json")
STANCE_FILE_MATCHERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (SOURCE_VOTING_RECORD, VOTING_ISSUE_FILE_SUFFIXES),
    (SOURCE_PUBLIC_RELEASE, PUBLIC_ISSUE_FILE_SUFFIXES),
)
STANCE_SOURCE_ORDER: tuple[str, ...] = (SOURCE_PUBLIC_RELEASE, SOURCE_VOTING_RECORD)
STANCE_SOURCE_LABELS = {
    SOURCE_PUBLIC_RELEASE: "Public release stance object",
    SOURCE_VOTING_RECORD: "Voting issue profile object",
}
ALL_STANCE_FILE_SUFFIXES: tuple[str, ...] = PUBLIC_ISSUE_FILE_SUFFIXES + VOTING_ISSUE_FILE_SUFFIXES
PARTY_FILTER_OPTIONS = ["All parties", "Republican", "Democrat", "Third party"]
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
TOPIC_RANK = {topic: idx for idx, topic in enumerate(TOPIC_ORDER)}
ARCHIA_DEFAULT_BASE_URL = "https://api.archia.app/v1"
LLM_MODEL_OPTIONS: tuple[tuple[str, str], ...] = (
    ("GPT 5.2", "gpt-5.2"),
    ("Claude Sonnet 4.5", "priv-claude-sonnet-4-5-20250929"),
)
DEFAULT_LLM_PROMPT = (
    "Identify potential contradictions between public-release stances and voting stances. "
    "Call out alignments and data gaps."
)


def load_json(path: Path) -> Any:
    preloaded = get_preloaded_json(path)
    if preloaded is not None:
        return preloaded
    return json.loads(path.read_text(encoding="utf-8"))


def read_env() -> None:
    project_root = Path(__file__).resolve().parents[2]
    env_path = project_root / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)
    else:
        load_dotenv(override=False)


def archia_response_text(response: Any) -> str:
    if isinstance(response, dict):
        output = response.get("output", [])
        if output and isinstance(output, list):
            content = output[0].get("content", [])
            if content and isinstance(content, list):
                text = content[0].get("text")
                if isinstance(text, str) and text.strip():
                    return text.strip()
    return json.dumps(response, ensure_ascii=False)


def run_archia_response(model_ref: str, prompt: str) -> str:
    read_env()
    archia_key = os.getenv("ARCHIA")
    if not archia_key:
        raise RuntimeError("Missing ARCHIA in .env or environment.")

    base_url = os.getenv("ARCHIA_BASE_URL", ARCHIA_DEFAULT_BASE_URL).rstrip("/")
    headers = {
        "x-api-key": archia_key.strip(),
        "Authorization": f"Bearer {archia_key.strip()}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_ref,
        "input": prompt,
    }

    with httpx.Client(http2=True, timeout=90, headers=headers) as client:
        response = client.post(f"{base_url}/responses", json=payload)
        response.raise_for_status()
        return archia_response_text(response.json())


def build_llm_prompt(
    *,
    member_name: str,
    bioguide_id: str,
    public_issue_profile: dict[str, Any],
    voting_issue_profile: dict[str, Any],
    user_prompt: str,
) -> str:
    request_payload = {
        "member": {
            "name": member_name,
            "bioguideId": bioguide_id,
        },
        "public_issue_profile": public_issue_profile,
        "voting_issue_profile": voting_issue_profile,
    }
    instruction = user_prompt.strip() or DEFAULT_LLM_PROMPT
    return (
        "You are helping with congressional hypocrisy analysis.\n"
        "Compare public_issue_profile vs voting_issue_profile for this member.\n"
        "If voting_issue_profile is empty, explain what cannot yet be inferred.\n"
        "Respond in Markdown with sections: Overall Assessment, Potential Contradictions, Alignments, Data Gaps.\n\n"
        f"User request: {instruction}\n\n"
        "Stance objects JSON:\n"
        f"{json.dumps(request_payload, ensure_ascii=False, indent=2)}"
    )


def render_llm_sidebar(selected_profile: dict[str, Any] | None) -> None:
    model_labels = [label for label, _ in LLM_MODEL_OPTIONS]
    model_map = {label: model for label, model in LLM_MODEL_OPTIONS}

    with st.sidebar:
        st.markdown("---")
        st.markdown("### LLM Analysis")
        selected_model_label = st.selectbox(
            "Model",
            model_labels,
            index=0,
            key="stances_llm_model_label",
        )
        user_prompt = st.text_area(
            "Prompt",
            value=DEFAULT_LLM_PROMPT,
            key="stances_llm_prompt",
            height=100,
        )

        if selected_profile is None:
            st.caption("Select one member in Navigator to enable LLM analysis.")
            st.button("Run LLM Analysis", use_container_width=True, disabled=True)
            return

        bioguide_id = str(selected_profile.get("bioguideId") or "")
        member_name = str(selected_profile.get("name") or bioguide_id or "-")
        st.caption(f"Member: {member_name} ({bioguide_id})")

        stances_map = selected_profile.get("stances") if isinstance(selected_profile.get("stances"), dict) else {}
        public_payload = stances_map.get(SOURCE_PUBLIC_RELEASE) if isinstance(stances_map.get(SOURCE_PUBLIC_RELEASE), dict) else {}
        voting_payload = stances_map.get(SOURCE_VOTING_RECORD) if isinstance(stances_map.get(SOURCE_VOTING_RECORD), dict) else {}
        public_issue_profile = public_payload.get("issues") if isinstance(public_payload.get("issues"), dict) else {}
        voting_issue_profile = voting_payload.get("issues") if isinstance(voting_payload.get("issues"), dict) else {}

        run_clicked = st.button("Run LLM Analysis", use_container_width=True)
        if run_clicked:
            prompt = build_llm_prompt(
                member_name=member_name,
                bioguide_id=bioguide_id,
                public_issue_profile=public_issue_profile,
                voting_issue_profile=voting_issue_profile,
                user_prompt=user_prompt,
            )
            with st.spinner("Querying Archia..."):
                try:
                    model_ref = model_map[selected_model_label]
                    response_text = run_archia_response(model_ref=model_ref, prompt=prompt)
                    st.session_state["stances_llm_result"] = response_text
                    st.session_state["stances_llm_error"] = ""
                    st.session_state["stances_llm_member_id"] = bioguide_id
                    st.session_state["stances_llm_model_ref"] = model_ref
                except Exception as exc:
                    st.session_state["stances_llm_result"] = ""
                    st.session_state["stances_llm_error"] = str(exc)
                    st.session_state["stances_llm_member_id"] = bioguide_id
                    st.session_state["stances_llm_model_ref"] = model_map[selected_model_label]

        last_member_id = str(st.session_state.get("stances_llm_member_id") or "")
        current_model_ref = model_map[selected_model_label]
        last_model_ref = str(st.session_state.get("stances_llm_model_ref") or "")
        if last_member_id != bioguide_id or last_model_ref != current_model_ref:
            return

        error_message = str(st.session_state.get("stances_llm_error") or "").strip()
        response_text = str(st.session_state.get("stances_llm_result") or "").strip()
        if error_message:
            st.error(error_message)
        if response_text:
            st.markdown("#### Response")
            st.markdown(response_text)

        with st.expander("Current stance payload", expanded=False):
            st.json(
                {
                    "public_issue_profile": public_issue_profile,
                    "voting_issue_profile": voting_issue_profile,
                },
                expanded=False,
            )


def to_int(value: Any, fallback: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def source_display_name(source_key: str) -> str:
    return STANCE_SOURCE_LABELS.get(source_key, source_key)


def parse_stance_filename(file_name: str) -> tuple[str, str] | None:
    for source_key, suffixes in STANCE_FILE_MATCHERS:
        matched_suffix = next((suffix for suffix in suffixes if file_name.endswith(suffix)), None)
        if matched_suffix is None:
            continue
        bioguide_id = file_name[: -len(matched_suffix)].strip()
        if not bioguide_id:
            return None
        return bioguide_id, source_key
    return None


def ordered_source_keys(stances: dict[str, Any]) -> list[str]:
    ordered = [source for source in STANCE_SOURCE_ORDER if source in stances]
    ordered.extend(sorted(source for source in stances if source not in ordered))
    return ordered


def topic_display_name(topic: str) -> str:
    return topic.replace("_", " ").title()


def party_bucket(party_name: str) -> str:
    normalized_party = normalize_text(party_name)
    if "republican" in normalized_party:
        return "Republican"
    if "democrat" in normalized_party:
        return "Democrat"
    return "Third party"


def member_label(profile: dict[str, Any]) -> str:
    name = str(profile.get("name") or profile.get("bioguideId") or "-")
    party = str(profile.get("partyBucket") or "-")
    state = str(profile.get("state") or "-")
    bioguide_id = str(profile.get("bioguideId") or "-")
    return f"{name} ({party}, {state}) [{bioguide_id}]"


def inject_stances_page_css() -> None:
    st.markdown(
        """
        <style>
          /* Closed select control */
          [data-testid="stSelectbox"] div[data-baseweb="select"] > div {
            background: #0f0f0f !important;
            border: 1px solid #444444 !important;
          }
          [data-testid="stSelectbox"] div[data-baseweb="select"] *,
          [data-testid="stSelectbox"] div[data-baseweb="select"] svg {
            color: #ffffff !important;
            fill: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
          }
          [data-testid="stSelectbox"] label {
            color: #ffffff !important;
          }

          /* Open dropdown/popup menu (portal-rendered) */
          div[data-baseweb="popover"],
          div[data-baseweb="menu"],
          ul[role="listbox"],
          div[role="option"] {
            background: #0f0f0f !important;
            color: #ffffff !important;
            border-color: #444444 !important;
          }
          div[data-baseweb="popover"] * ,
          div[data-baseweb="menu"] * {
            color: #ffffff !important;
            fill: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
          }
          div[data-baseweb="popover"] > div,
          div[data-baseweb="popover"] > div > div,
          div[data-baseweb="menu"] > div,
          div[data-baseweb="menu"] > div > div {
            background: #0f0f0f !important;
            color: #ffffff !important;
          }
          [role="listbox"],
          [role="listbox"] * {
            background: #0f0f0f !important;
            color: #ffffff !important;
            fill: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
          }
          ul[role="listbox"] li,
          ul[role="listbox"] li > div,
          li[role="option"],
          li[role="option"] > div,
          div[role="option"],
          div[role="option"] > div {
            background: #0f0f0f !important;
            color: #ffffff !important;
            fill: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
          }
          li[role="option"] *,
          div[role="option"] * {
            color: #ffffff !important;
            fill: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
          }
          ul[role="listbox"] li[aria-selected="true"],
          ul[role="listbox"] li:hover,
          li[role="option"][aria-selected="true"],
          li[role="option"]:hover,
          div[role="option"][aria-selected="true"],
          div[role="option"]:hover {
            background: #212121 !important;
            color: #ffffff !important;
          }
          [data-testid="stExpander"] summary {
            font-size: 1.05rem !important;
            font-weight: 700 !important;
          }
          .issue-heading {
            font-size: 1.22rem;
            font-weight: 800;
            line-height: 1.3;
            margin-bottom: 0.35rem;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


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


def build_profile_search_text(profile: dict[str, Any], issues: dict[str, dict[str, Any]]) -> str:
    parts = [
        str(profile.get("name") or ""),
        str(profile.get("bioguideId") or ""),
        str(profile.get("party") or ""),
        str(profile.get("partyBucket") or ""),
        str(profile.get("state") or ""),
    ]
    for topic in TOPIC_ORDER:
        issue_payload = issues.get(topic) if isinstance(issues.get(topic), dict) else {}
        parts.append(topic)
        parts.append(str(issue_payload.get("summary") or "")[:240])
    return normalize_text(" ".join(parts))


def build_issue_rows(profile: dict[str, Any], issues: dict[str, dict[str, Any]], source_key: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for topic in TOPIC_ORDER:
        issue_payload = issues.get(topic) if isinstance(issues.get(topic), dict) else {}
        summary = str(issue_payload.get("summary") or "").strip()
        evidence = max(to_int(issue_payload.get("evidence"), 0), 0)
        rows.append(
            {
                "memberName": str(profile.get("name") or profile.get("bioguideId") or "-"),
                "bioguideId": str(profile.get("bioguideId") or "-"),
                "topic": topic,
                "topicLabel": topic_display_name(topic),
                "sourceKey": source_key,
                "sourceLabel": source_display_name(source_key),
                "summary": summary,
                "evidence": evidence,
                "_search_text": normalize_text(
                    " ".join(
                        [
                            str(profile.get("name") or ""),
                            str(profile.get("bioguideId") or ""),
                            topic,
                            topic_display_name(topic),
                            source_display_name(source_key),
                            summary[:600],
                        ]
                    )
                ),
            }
        )
    return rows


def render_source_issue_panel(
    source_key: str,
    source_payload: dict[str, Any],
    query: str,
    search_scope: str,
    expand_all: bool,
) -> None:
    issue_rows = list(source_payload.get("issueRows") or [])
    total_evidence = int(source_payload.get("totalEvidence") or 0)
    issues_with_evidence = int(source_payload.get("issuesWithEvidence") or 0)

    st.markdown(f"#### {source_display_name(source_key)}")
    st.caption(f"Source file: {source_payload.get('sourceFile') or '-'}")

    c1, c2, c3 = st.columns(3)
    c1.metric("Issue fields", len(issue_rows))
    c2.metric("With evidence", issues_with_evidence)
    c3.metric("Total evidence", total_evidence)

    matched_topic_set: set[str] = set()
    if query.strip() and search_scope == "Current selection":
        issue_search_base = list(range(len(issue_rows)))
        issue_search_texts = [str(row.get("_search_text") or "") for row in issue_rows]
        matched_issue_indices = fuzzy_filter_indices(issue_search_base, issue_search_texts, query, 0.42)
        matched_topic_set = {str(issue_rows[idx].get("topic") or "") for idx in matched_issue_indices}
        issue_rows = sorted(
            issue_rows,
            key=lambda row: (
                str(row.get("topic") or "") not in matched_topic_set,
                TOPIC_RANK.get(str(row.get("topic") or ""), 999),
            ),
        )
        st.caption(f"Fuzzy query matched {len(matched_topic_set)} issue field(s) in this stance object.")

    for row in issue_rows:
        topic_label = str(row.get("topicLabel") or "Unknown issue")
        evidence_count = int(row.get("evidence") or 0)
        summary = str(row.get("summary") or "").strip()
        topic_key = str(row.get("topic") or "")
        match_hint = ""
        if query.strip() and search_scope == "Current selection":
            match_hint = "Matched query" if topic_key in matched_topic_set else "No query match"

        title_bits = [f"{topic_label}", f"{evidence_count} evidence"]
        if match_hint:
            title_bits.append(match_hint)
        expander_title = " | ".join(title_bits)

        with st.expander(expander_title, expanded=expand_all):
            st.markdown(f"<div class='issue-heading'>{topic_label}</div>", unsafe_allow_html=True)
            if evidence_count > 0:
                st.caption(f"Evidence relationship: {evidence_count} supporting source item(s) linked to this issue.")
            else:
                st.caption("Evidence relationship: no supporting source items were linked to this issue.")
            st.markdown("**Summary**")
            if summary:
                st.write(summary)
            else:
                st.caption("No summary extracted for this issue.")


@st.cache_data(show_spinner=False)
def load_issue_profiles(stances_dir_path: str, signature: str) -> list[dict[str, Any]]:
    del signature
    stances_dir = Path(stances_dir_path)
    if not stances_dir.exists():
        return []

    profiles_by_member: dict[str, dict[str, Any]] = {}
    for issue_file in sorted(stances_dir.glob("*.json")):
        parsed = parse_stance_filename(issue_file.name)
        if parsed is None:
            continue

        bioguide_id, source_key = parsed

        try:
            payload = load_json(issue_file)
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue

        normalized_issues: dict[str, dict[str, Any]] = {}
        for topic in TOPIC_ORDER:
            topic_payload = payload.get(topic) if isinstance(payload.get(topic), dict) else {}
            normalized_issues[topic] = {
                "summary": str(topic_payload.get("summary") or "").strip(),
                "evidence": max(to_int(topic_payload.get("evidence"), 0), 0),
            }

        member_entry = profiles_by_member.setdefault(
            bioguide_id,
            {
                "bioguideId": bioguide_id,
                "stances": {},
            },
        )
        stances = member_entry["stances"]
        existing = stances.get(source_key) if isinstance(stances, dict) else None
        existing_mtime = int(existing.get("_mtime_ns") or 0) if isinstance(existing, dict) else 0
        candidate_mtime = int(issue_file.stat().st_mtime_ns)
        if candidate_mtime < existing_mtime:
            continue

        stances[source_key] = {
            "issues": normalized_issues,
            "sourceFile": issue_file.name,
            "_mtime_ns": candidate_mtime,
        }

    profiles: list[dict[str, Any]] = []
    for bioguide_id in sorted(profiles_by_member):
        member_entry = profiles_by_member[bioguide_id]
        stances = member_entry.get("stances") if isinstance(member_entry.get("stances"), dict) else {}
        if not stances:
            continue

        cleaned_stances: dict[str, dict[str, Any]] = {}
        for source_key in ordered_source_keys(stances):
            payload = stances.get(source_key)
            if not isinstance(payload, dict):
                continue
            cleaned_stances[source_key] = {
                "issues": payload.get("issues") if isinstance(payload.get("issues"), dict) else {},
                "sourceFile": str(payload.get("sourceFile") or "-"),
            }
        profiles.append(
            {
                "bioguideId": bioguide_id,
                "stances": cleaned_stances,
            }
        )
    return profiles


@st.cache_data(show_spinner=False)
def load_member_metadata(press_releases_path: str, mtime_ns: int) -> dict[str, dict[str, Any]]:
    del mtime_ns
    payload = load_json(Path(press_releases_path))
    members = payload.get("membersByBioguideId") if isinstance(payload, dict) else {}
    if not isinstance(members, dict):
        return {}

    metadata: dict[str, dict[str, Any]] = {}
    for bioguide_id, member_data in members.items():
        if not isinstance(member_data, dict):
            continue
        party_name = str(member_data.get("partyName") or "-")
        metadata[str(bioguide_id)] = {
            "name": str(member_data.get("name") or bioguide_id),
            "party": party_name,
            "partyBucket": party_bucket(party_name),
            "state": str(member_data.get("state") or "-"),
            "releaseCount": max(to_int(member_data.get("releaseCount"), 0), 0),
            "status": str(member_data.get("status") or ""),
        }
    return metadata


def main() -> None:
    inject_css()
    inject_stances_page_css()

    st.markdown(
        """
        <div class="hero">
          <p class="hero-title">Stances</p>
          <p class="hero-sub">Browse per-member issue profiles and evidence counts from extracted press release stances.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    stance_files = [
        p
        for p in sorted(STANCES_DIR.glob("*.json"))
        if parse_stance_filename(p.name) is not None
    ] if STANCES_DIR.exists() else []
    if not stance_files:
        st.error(f"No issue profile files found in {STANCES_DIR} with suffixes: {', '.join(ALL_STANCE_FILE_SUFFIXES)}")
        return

    stance_signature = "|".join(f"{p.name}:{p.stat().st_mtime_ns}" for p in stance_files)
    all_profiles = load_issue_profiles(str(STANCES_DIR), stance_signature)
    if not all_profiles:
        st.info("No issue profiles were loaded from the stances directory.")
        return

    member_metadata: dict[str, dict[str, Any]] = {}
    if PRESS_RELEASES_FILE.exists():
        member_metadata = load_member_metadata(str(PRESS_RELEASES_FILE), PRESS_RELEASES_FILE.stat().st_mtime_ns)
    else:
        st.warning(f"Member metadata file not found: {PRESS_RELEASES_FILE}")

    enriched_profiles: list[dict[str, Any]] = []
    for profile in all_profiles:
        bioguide_id = str(profile.get("bioguideId") or "")
        meta = member_metadata.get(bioguide_id, {})
        party_name = str(meta.get("party") or "Unknown")
        enriched = {
            **profile,
            "name": str(meta.get("name") or bioguide_id),
            "party": party_name,
            "partyBucket": str(meta.get("partyBucket") or party_bucket(party_name)),
            "state": str(meta.get("state") or "-"),
            "pressReleaseCount": max(to_int(meta.get("releaseCount"), 0), 0),
            "pressReleaseStatus": str(meta.get("status") or "unknown"),
            "stances": {},
        }
        stances = profile.get("stances") if isinstance(profile.get("stances"), dict) else {}
        source_search_texts: dict[str, str] = {}
        for source_key in ordered_source_keys(stances):
            source_payload = stances.get(source_key) if isinstance(stances.get(source_key), dict) else {}
            issues = source_payload.get("issues") if isinstance(source_payload.get("issues"), dict) else {}
            issue_rows = build_issue_rows(enriched, issues, source_key)
            search_text = build_profile_search_text(enriched, issues)
            enriched["stances"][source_key] = {
                "issues": issues,
                "sourceFile": str(source_payload.get("sourceFile") or "-"),
                "issueRows": issue_rows,
                "issuesWithEvidence": sum(1 for row in issue_rows if int(row.get("evidence") or 0) > 0),
                "totalEvidence": sum(int(row.get("evidence") or 0) for row in issue_rows),
                "_search_text": search_text,
            }
            source_search_texts[source_key] = search_text

        default_source = next((source for source in STANCE_SOURCE_ORDER if source in enriched["stances"]), "")
        if not default_source and enriched["stances"]:
            default_source = next(iter(enriched["stances"]))
        enriched["defaultSource"] = default_source
        default_source_payload = enriched["stances"].get(default_source) if default_source else {}
        default_issue_rows = default_source_payload.get("issueRows") if isinstance(default_source_payload, dict) else []
        enriched["issueRows"] = list(default_issue_rows or [])
        enriched["issuesWithEvidence"] = int(default_source_payload.get("issuesWithEvidence") or 0) if isinstance(default_source_payload, dict) else 0
        enriched["totalEvidence"] = int(default_source_payload.get("totalEvidence") or 0) if isinstance(default_source_payload, dict) else 0
        enriched["_search_text"] = normalize_text(" ".join(source_search_texts.values()))
        enriched_profiles.append(enriched)

    enriched_profiles.sort(key=lambda item: (str(item.get("name") or ""), str(item.get("bioguideId") or "")))
    all_states = sorted({str(profile.get("state") or "-") for profile in enriched_profiles})

    st.subheader("Navigator")
    nav1, nav2, nav3, nav4, nav5 = st.columns([2.0, 1.2, 1.0, 1.3, 1.2])
    with nav2:
        selected_party = st.selectbox("Party", PARTY_FILTER_OPTIONS, index=0)
    with nav3:
        selected_state = st.selectbox("State", ["All states"] + all_states, index=0)
    with nav4:
        search_scope = st.selectbox(
            "Search scope",
            ["Current selection", "All members"],
            index=0,
            help="Use 'All members' to search every member in the current Party/State context.",
        )
    with nav5:
        query = st.text_input("Fuzzy search", "", placeholder="Try: border security")

    filtered_profiles: list[dict[str, Any]] = []
    for profile in enriched_profiles:
        party_value = str(profile.get("partyBucket") or "Third party")
        state_value = str(profile.get("state") or "-")
        party_ok = selected_party == "All parties" or party_value == selected_party
        state_ok = selected_state == "All states" or state_value == selected_state
        if party_ok and state_ok:
            filtered_profiles.append(profile)

    if not filtered_profiles:
        st.warning("No members match Party/State filters.")
        return

    profile_map = {str(profile.get("bioguideId") or ""): profile for profile in filtered_profiles}
    member_options = ["ALL"] + [str(profile.get("bioguideId") or "") for profile in filtered_profiles]

    selected_member_state_key = "stances_selected_member_id"
    selected_member_id = str(st.session_state.get(selected_member_state_key) or "ALL")
    if selected_member_id not in member_options:
        st.session_state[selected_member_state_key] = "ALL"

    with nav1:
        st.selectbox(
            "Search/select member",
            member_options,
            key=selected_member_state_key,
            format_func=lambda member_id: "All members" if member_id == "ALL" else member_label(profile_map[member_id]),
            help="Type in this box to autocomplete by member name or bioguideId.",
        )
    selected_member_id = str(st.session_state.get(selected_member_state_key) or "ALL")
    render_llm_sidebar(profile_map.get(selected_member_id) if selected_member_id != "ALL" else None)

    search_base_profiles = filtered_profiles
    if search_scope == "Current selection" and selected_member_id != "ALL":
        search_base_profiles = [profile_map[selected_member_id]]

    base_indices = list(range(len(search_base_profiles)))
    search_texts = [str(profile.get("_search_text") or "") for profile in search_base_profiles]
    matched_profile_indices = fuzzy_filter_indices(base_indices, search_texts, query, 0.42)
    matched_profiles = [search_base_profiles[idx] for idx in matched_profile_indices]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Stance files", len(stance_files))
    c2.metric("Members loaded", len(enriched_profiles))
    c3.metric("Members in context", len(filtered_profiles))
    c4.metric("Fuzzy member matches", len(matched_profiles))

    if selected_member_id == "ALL":
        st.markdown("### Members")
        st.caption(
            "Select one member in the navigator to inspect all static issue fields in collapsible detail panels."
        )
        preview_profiles = matched_profiles if query.strip() else filtered_profiles
        preview_rows = [
            {
                "Member": str(profile.get("name") or "-"),
                "Bioguide ID": str(profile.get("bioguideId") or "-"),
                "Party": str(profile.get("partyBucket") or "-"),
                "State": str(profile.get("state") or "-"),
                "Issue fields with evidence": int(profile.get("issuesWithEvidence") or 0),
                "Total evidence": int(profile.get("totalEvidence") or 0),
            }
            for profile in preview_profiles
        ]
        st.dataframe(preview_rows, hide_index=True, width="stretch")
        return

    selected_profile = profile_map[selected_member_id]
    stances_map = selected_profile.get("stances") if isinstance(selected_profile.get("stances"), dict) else {}
    source_options = ordered_source_keys(stances_map)
    if not source_options:
        st.warning("No stance objects found for the selected member.")
        return

    st.markdown("### Member")
    st.markdown(f"**{selected_profile.get('name') or '-'}**")
    st.caption(
        f"Bioguide ID: {selected_profile.get('bioguideId') or '-'} | "
        f"Party: {selected_profile.get('partyBucket') or '-'} | "
        f"State: {selected_profile.get('state') or '-'}"
    )
    st.caption(
        f"Stance objects loaded: {len(source_options)} | "
        f"Press releases: {int(selected_profile.get('pressReleaseCount') or 0)} | "
        f"Press release metadata status: {selected_profile.get('pressReleaseStatus') or 'unknown'}"
    )

    st.markdown("### Stance Objects")
    st.caption(
        "Each issue field is static across members. Evidence is the number of source items supporting that issue summary. "
        "If both public-release and voting stance objects exist, they are shown side by side."
    )
    expand_all = st.checkbox(
        "Expand all issues for current member",
        value=False,
        help="Open all issue panels for this member's visible stance object(s).",
        key=f"stances_expand_all_{selected_member_id}",
    )
    if query.strip() and search_scope == "All members":
        st.caption(
            f"Fuzzy query matched {len(matched_profiles)} member profile(s) across the current Party/State context."
        )

    ordered_sources: list[str] = []
    for source_key in (SOURCE_PUBLIC_RELEASE, SOURCE_VOTING_RECORD):
        if source_key in source_options:
            ordered_sources.append(source_key)
    for source_key in source_options:
        if source_key not in ordered_sources:
            ordered_sources.append(source_key)

    has_public_and_voting = SOURCE_PUBLIC_RELEASE in ordered_sources and SOURCE_VOTING_RECORD in ordered_sources
    if has_public_and_voting:
        left_col, right_col = st.columns(2)
        with left_col:
            render_source_issue_panel(
                SOURCE_PUBLIC_RELEASE,
                stances_map.get(SOURCE_PUBLIC_RELEASE) if isinstance(stances_map.get(SOURCE_PUBLIC_RELEASE), dict) else {},
                query,
                search_scope,
                expand_all,
            )
        with right_col:
            render_source_issue_panel(
                SOURCE_VOTING_RECORD,
                stances_map.get(SOURCE_VOTING_RECORD) if isinstance(stances_map.get(SOURCE_VOTING_RECORD), dict) else {},
                query,
                search_scope,
                expand_all,
            )
        extra_sources = [source for source in ordered_sources if source not in {SOURCE_PUBLIC_RELEASE, SOURCE_VOTING_RECORD}]
        for source_key in extra_sources:
            st.markdown("---")
            render_source_issue_panel(
                source_key,
                stances_map.get(source_key) if isinstance(stances_map.get(source_key), dict) else {},
                query,
                search_scope,
                expand_all,
            )
    else:
        for source_key in ordered_sources:
            render_source_issue_panel(
                source_key,
                stances_map.get(source_key) if isinstance(stances_map.get(source_key), dict) else {},
                query,
                search_scope,
                expand_all,
            )

    if query.strip() and search_scope == "All members":
        st.markdown("### Query Matches Across Members")
        issue_records: list[dict[str, Any]] = []
        for profile in filtered_profiles:
            profile_stances = profile.get("stances") if isinstance(profile.get("stances"), dict) else {}
            for source_key in ordered_source_keys(profile_stances):
                payload = profile_stances.get(source_key) if isinstance(profile_stances.get(source_key), dict) else {}
                issue_records.extend(list(payload.get("issueRows") or []))

        issue_base_indices = list(range(len(issue_records)))
        issue_search_texts = [str(row.get("_search_text") or "") for row in issue_records]
        matched_issue_indices = fuzzy_filter_indices(issue_base_indices, issue_search_texts, query, 0.42)
        matched_issue_records = [issue_records[idx] for idx in matched_issue_indices]

        if not matched_issue_records:
            st.info("No issue rows matched the fuzzy search across current Party/State members.")
        else:
            max_preview = 60
            preview_rows = []
            for row in matched_issue_records[:max_preview]:
                preview_rows.append(
                    {
                        "Member": str(row.get("memberName") or "-"),
                        "Bioguide ID": str(row.get("bioguideId") or "-"),
                        "Source": str(row.get("sourceLabel") or "-"),
                        "Issue": str(row.get("topicLabel") or "-"),
                        "Evidence": int(row.get("evidence") or 0),
                        "Summary preview": str(row.get("summary") or "")[:220],
                    }
                )
            st.caption(f"Showing {len(preview_rows)} of {len(matched_issue_records)} matched issue rows.")
            st.dataframe(preview_rows, hide_index=True, width="stretch")


if __name__ == "__main__":
    main()
