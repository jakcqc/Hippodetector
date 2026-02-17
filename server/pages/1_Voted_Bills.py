from __future__ import annotations

import json
import math
from pathlib import Path
import sys
from typing import Any

from rapidfuzz import fuzz, process
import streamlit as st

try:
    from app import HIPPO_ICON_PATH, inject_css, normalize_text, strip_html
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from app import HIPPO_ICON_PATH, inject_css, normalize_text, strip_html


DATA_DIR = Path(__file__).resolve().parents[2] / "data"
VOTED_BILLS_FILE = DATA_DIR / "congress_bills_voted_compact_last_1_year.json"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def inject_votes_page_css() -> None:
    st.markdown(
        """
        <style>
          /* Keep dataframe controls readable in darker toolbars */
          [data-testid="stDataFrame"] button svg,
          [data-testid="stDataFrame"] svg,
          [data-testid="stDataFrame"] [role="button"] svg {
            fill: #ffffff !important;
            color: #ffffff !important;
          }
          [data-testid="stDataFrame"] button {
            color: #ffffff !important;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def bill_search_text(bill: dict[str, Any]) -> str:
    title = str(bill.get("title") or "")
    bill_id = str(bill.get("billId") or "")
    bill_type = str(bill.get("billType") or "")
    bill_number = str(bill.get("billNumber") or "")
    summaries = bill.get("summaries") or []
    summary_text = ""
    if isinstance(summaries, list) and summaries:
        first_summary = summaries[0]
        if isinstance(first_summary, dict):
            summary_text = strip_html(str(first_summary.get("text") or ""))[:900]
    votes = bill.get("votes") or []
    vote_blurbs = []
    if isinstance(votes, list):
        for vote in votes[:5]:
            if isinstance(vote, dict):
                vote_blurbs.append(
                    " ".join(
                        [
                            str(vote.get("chamber") or ""),
                            str(vote.get("question") or ""),
                            str(vote.get("result") or ""),
                            str(vote.get("voteDate") or ""),
                        ]
                    )
                )
    return normalize_text(" ".join([title, bill_id, bill_type, bill_number, summary_text, " ".join(vote_blurbs)]))


@st.cache_data(show_spinner=False)
def build_bill_index(data_path: str, mtime_ns: int) -> dict[str, Any]:
    del mtime_ns
    payload = load_json(Path(data_path))
    bills = payload.get("bills", []) if isinstance(payload, dict) else []
    normalized_bills = [bill for bill in bills if isinstance(bill, dict)]
    search_texts = [bill_search_text(bill) for bill in normalized_bills]
    sort_index_map: dict[str, list[int]] = {}
    for sort_field in ["Latest update", "Vote count", "Version count", "Bill ID", "Title"]:
        sort_index_map[f"{sort_field}:asc"] = build_sorted_indices(normalized_bills, sort_field, False)
        sort_index_map[f"{sort_field}:desc"] = build_sorted_indices(normalized_bills, sort_field, True)
    return {"payload": payload, "bills": normalized_bills, "search_texts": search_texts, "sort_index_map": sort_index_map}


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


def extract_proposer_parties(bill: dict[str, Any]) -> list[str]:
    values = bill.get("proposerParties")
    if isinstance(values, list):
        cleaned = [str(v).strip() for v in values if str(v).strip()]
        return cleaned or ["Unknown"]
    return ["Unknown"]


def extract_proposer_states(bill: dict[str, Any]) -> list[str]:
    values = bill.get("proposerStates")
    if isinstance(values, list):
        cleaned = [str(v).strip() for v in values if str(v).strip()]
        return cleaned or ["Unknown"]
    return ["Unknown"]


def vote_outcome_label(vote: dict[str, Any]) -> str:
    result = normalize_text(str(vote.get("result") or ""))
    if any(token in result for token in ["passed", "agreed", "adopted", "confirmed", "ratified", "approved"]):
        return "Passed"
    if any(token in result for token in ["defeated", "rejected", "failed", "not passed", "not agreed"]):
        return "Not Passed"
    return "Unknown"


def bill_status_label(bill: dict[str, Any]) -> str:
    votes = bill.get("votes") or []
    labels = {vote_outcome_label(v) for v in votes if isinstance(v, dict)}
    labels.discard("Unknown")
    if not labels:
        return "Unknown"
    if len(labels) > 1:
        return "Mixed"
    return next(iter(labels))


def build_sorted_indices(
    bills: list[dict[str, Any]],
    sort_field: str,
    descending: bool,
) -> list[int]:
    def sort_key(item: tuple[int, dict[str, Any]]) -> tuple[Any, ...]:
        idx, bill = item
        if sort_field == "Latest update":
            return (str(bill.get("latestUpdateDate") or ""), idx)
        if sort_field == "Vote count":
            return (int(bill.get("voteCount") or len(bill.get("votes") or [])), idx)
        if sort_field == "Version count":
            return (int(bill.get("versionCount") or 1), idx)
        if sort_field == "Bill ID":
            return (str(bill.get("billId") or ""), idx)
        return (str(bill.get("title") or ""), idx)

    ranked = sorted(list(enumerate(bills)), key=sort_key, reverse=descending)
    return [idx for idx, _ in ranked]


def main() -> None:
    page_icon: str | Path = "🦛"
    if HIPPO_ICON_PATH.exists():
        page_icon = HIPPO_ICON_PATH
    st.set_page_config(page_title="Hippodetector Voted Bills", page_icon=page_icon, layout="wide")
    inject_css()
    inject_votes_page_css()

    st.markdown(
        """
        <div class="hero">
          <p class="hero-title">Voted Bills Explorer</p>
          <p class="hero-sub">Search bills with recorded votes and inspect each vote event in detail.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not VOTED_BILLS_FILE.exists():
        st.error(
            f"Required file not found: {VOTED_BILLS_FILE}. "
            "Build it with `python dataset/build_voted_bills_compact.py`."
        )
        return

    dataset_mtime_ns = VOTED_BILLS_FILE.stat().st_mtime_ns
    index_data = build_bill_index(str(VOTED_BILLS_FILE), dataset_mtime_ns)
    payload = index_data["payload"]
    bills = index_data["bills"]
    search_texts = index_data["search_texts"]
    sort_index_map = index_data["sort_index_map"]

    if not bills:
        st.info("No voted bills found in compact dataset.")
        return

    chambers = sorted(
        {
            str(vote.get("chamber") or "")
            for bill in bills
            for vote in (bill.get("votes") or [])
            if isinstance(vote, dict) and vote.get("chamber")
        }
    )
    congresses = sorted({str(bill.get("congress") or "") for bill in bills if bill.get("congress")})
    proposer_parties = sorted({party for bill in bills for party in extract_proposer_parties(bill)})
    proposer_states = sorted({state for bill in bills for state in extract_proposer_states(bill)})
    status_options = ["All", "Passed", "Not Passed", "Mixed", "Unknown"]

    st.subheader("Navigator")
    nav1, nav2, nav3, nav4, nav5 = st.columns([2.1, 1.0, 1.0, 1.0, 1.0])
    with nav1:
        query = st.text_input("Fuzzy search", "", placeholder="Try: On Passage House border")
    with nav2:
        selected_chamber = st.selectbox("Chamber", ["Any chamber"] + chambers, index=0)
    with nav3:
        selected_congress = st.selectbox("Congress", ["All"] + congresses, index=0)
    with nav4:
        selected_status = st.selectbox("Passed status", status_options, index=0)
    with nav5:
        page_size = st.number_input("Bills/page", min_value=1, max_value=250, value=25, step=5)
    nav6, nav7, nav8, nav9 = st.columns([1.0, 1.0, 1.2, 1.0])
    with nav6:
        selected_party = st.selectbox("Proposer party", ["All"] + proposer_parties, index=0)
    with nav7:
        selected_state = st.selectbox("Proposer state", ["All"] + proposer_states, index=0)
    with nav8:
        sort_field = st.selectbox(
            "Sort by",
            ["Latest update", "Vote count", "Version count", "Bill ID", "Title"],
            index=0,
        )
    with nav9:
        sort_order = st.selectbox("Order", ["Descending", "Ascending"], index=0)

    sort_key = f"{sort_field}:{'desc' if sort_order == 'Descending' else 'asc'}"
    sorted_indices = sort_index_map.get(sort_key, list(range(len(bills))))
    base_indices = sorted_indices
    filtered_indices = []
    for idx in base_indices:
        bill = bills[idx]
        congress_ok = selected_congress == "All" or str(bill.get("congress") or "") == selected_congress
        chamber_ok = True
        if selected_chamber != "Any chamber":
            chamber_ok = any(
                isinstance(vote, dict) and str(vote.get("chamber") or "") == selected_chamber
                for vote in (bill.get("votes") or [])
            )
        status_ok = selected_status == "All" or bill_status_label(bill) == selected_status
        party_ok = selected_party == "All" or selected_party in extract_proposer_parties(bill)
        state_ok = selected_state == "All" or selected_state in extract_proposer_states(bill)
        if congress_ok and chamber_ok and status_ok and party_ok and state_ok:
            filtered_indices.append(idx)

    search_signature = (
        str(VOTED_BILLS_FILE),
        dataset_mtime_ns,
        selected_chamber,
        selected_congress,
        selected_status,
        selected_party,
        selected_state,
        sort_field,
        sort_order,
        normalize_text(query),
    )
    if st.session_state.get("voted_bill_search_signature") != search_signature:
        st.session_state["voted_bill_indices"] = fuzzy_filter_indices(filtered_indices, search_texts, query, 0.42)
        st.session_state["voted_bill_search_signature"] = search_signature
        st.session_state["voted_bill_page"] = 1

    matched_indices = st.session_state.get("voted_bill_indices", filtered_indices)

    total_results = len(matched_indices)
    total_pages = max(1, math.ceil(total_results / int(page_size)))
    current_page = int(st.session_state.get("voted_bill_page", 1))
    current_page = max(1, min(current_page, total_pages))
    st.session_state["voted_bill_page"] = current_page

    pager1, pager2, pager3, pager4 = st.columns([1, 1.2, 1, 1.2])
    with pager1:
        prev_clicked = st.button("Prev page", disabled=current_page <= 1, key="bills_prev")
    with pager2:
        page_input = int(
            st.number_input("Page", min_value=1, max_value=total_pages, value=current_page, step=1, key="bills_page_input")
        )
    with pager3:
        next_clicked = st.button("Next page", disabled=current_page >= total_pages, key="bills_next")
    with pager4:
        st.caption(f"Page {current_page} of {total_pages}")

    if prev_clicked and current_page > 1:
        current_page -= 1
    if next_clicked and current_page < total_pages:
        current_page += 1
    if not prev_clicked and not next_clicked:
        current_page = page_input
    current_page = max(1, min(current_page, total_pages))
    st.session_state["voted_bill_page"] = current_page

    start_idx = (current_page - 1) * int(page_size)
    end_idx = start_idx + int(page_size)
    paged_indices = matched_indices[start_idx:end_idx]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Voted Bills", str(payload.get("votedBillsCount", len(bills))))
    c2.metric("Bills Matched", total_results)
    c3.metric("Vote Events", str(payload.get("voteEventsCount", "-")))
    c4.metric("Versioned Bills", str(sum(1 for b in bills if int(b.get("versionCount", 1)) > 1)))

    if not paged_indices:
        st.info("No bills match current filters.")
        return

    st.caption(f"Showing bills {start_idx + 1}-{min(end_idx, total_results)} of {total_results}.")

    st.subheader("Bills")
    for local_idx, bill_idx in enumerate(paged_indices, start=0):
        global_idx = start_idx + local_idx
        bill = bills[bill_idx]
        bill_id = str(bill.get("billId") or "-")
        title = str(bill.get("title") or "Untitled bill")
        vote_count = int(bill.get("voteCount") or len(bill.get("votes") or []))
        version_count = int(bill.get("versionCount") or 1)
        header = f"{global_idx + 1}. {bill_id} - {title} | votes: {vote_count} | versions: {version_count}"

        with st.expander(header, expanded=(local_idx == 0)):
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown(f"<div class='field'><b>Bill:</b> {bill_id}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='field'><b>Title:</b> {title}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='field'><b>Congress:</b> {bill.get('congress', '-')}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='field'><b>Type/Number:</b> {bill.get('billType', '-')}-{bill.get('billNumber', '-')}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='field'><b>Latest update:</b> {bill.get('latestUpdateDate', '-')}</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            summaries = bill.get("summaries") or []
            if isinstance(summaries, list) and summaries:
                top_summary = summaries[0] if isinstance(summaries[0], dict) else {}
                summary_html = str(top_summary.get("text") or "")
                if summary_html:
                    st.markdown(f"<div class='formatted-body'>{summary_html}</div>", unsafe_allow_html=True)

            votes = bill.get("votes") or []
            if not isinstance(votes, list) or not votes:
                st.write("No votes found.")
                continue

            st.markdown("**Votes**")
            for vote_idx, vote in enumerate(votes, start=1):
                if not isinstance(vote, dict):
                    continue
                vote_header = (
                    f"Vote {vote_idx}: {vote.get('chamber', '-')} roll {vote.get('rollNumber', '-')} "
                    f"| {vote.get('question', '-')} | {vote.get('result', '-')}"
                )
                with st.expander(vote_header, expanded=False):
                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    st.markdown(f"<div class='field'><b>Date:</b> {vote.get('voteDate', '-')}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='field'><b>Recorded at:</b> {vote.get('recordedAt', '-')}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='field'><b>Question:</b> {vote.get('question', '-')}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='field'><b>Result:</b> {vote.get('result', '-')}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='field'><b>Source:</b> {vote.get('sourceUrl', '-')}</div>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)

                    vote_counts = vote.get("voteCounts") if isinstance(vote.get("voteCounts"), dict) else {}
                    if vote_counts:
                        vc1, vc2, vc3, vc4, vc5 = st.columns(5)
                        vc1.metric("Yea", int(vote_counts.get("yea", 0)))
                        vc2.metric("Nay", int(vote_counts.get("nay", 0)))
                        vc3.metric("Present", int(vote_counts.get("present", 0)))
                        vc4.metric("Not Voting", int(vote_counts.get("notVoting", 0)))
                        vc5.metric("Other", int(vote_counts.get("other", 0)))

                    members = vote.get("members") if isinstance(vote.get("members"), list) else []
                    if members:
                        rows = []
                        for member in members:
                            if not isinstance(member, dict):
                                continue
                            rows.append(
                                {
                                    "name": member.get("name"),
                                    "party": member.get("party"),
                                    "state": member.get("state"),
                                    "voteCast": member.get("voteCast"),
                                    "bioguideId": member.get("bioguideId"),
                                    "lisMemberId": member.get("lisMemberId"),
                                }
                            )
                        st.dataframe(rows, width="stretch", hide_index=True, height=280)


if __name__ == "__main__":
    main()
