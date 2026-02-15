from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime, timezone
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen


DEFAULT_MEMBERS_FILE = Path("data/congress_members.json")
DEFAULT_OUTPUT_PATH = Path("data/press_releases_by_bioguide.json")


class ListingParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.release_paths: Set[str] = set()
        self.next_href: Optional[str] = None

    def handle_starttag(self, tag: str, attrs: List[tuple[str, Optional[str]]]) -> None:
        if tag != "a":
            return

        attrs_dict = dict(attrs)
        href = attrs_dict.get("href") or ""
        rel = (attrs_dict.get("rel") or "").lower()

        if re.fullmatch(r"/media/press-releases/[^/?#]+/?", href):
            self.release_paths.add(href)

        if "next" in rel and href:
            self.next_href = href


class TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: List[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)

    def get_text(self) -> str:
        text = " ".join(part.strip() for part in self.parts if part.strip())
        return re.sub(r"\s+", " ", text).strip()


class DetailParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self.in_h1 = False
        self.in_date_div = False
        self.in_body = False
        self.body_div_depth = 0
        self.title_parts: List[str] = []
        self.date_parts: List[str] = []
        self.body_parts: List[str] = []

    def handle_starttag(self, tag: str, attrs: List[tuple[str, Optional[str]]]) -> None:
        attrs_dict = dict(attrs)
        classes = attrs_dict.get("class") or ""

        if tag == "h1":
            self.in_h1 = True

        if tag == "div" and "col-auto" in classes and not self.date_parts:
            self.in_date_div = True

        if tag == "div" and "evo-press-release__body" in classes and not self.in_body:
            self.in_body = True
            self.body_div_depth = 1
            return

        if self.in_body:
            if tag == "div":
                self.body_div_depth += 1
            self.body_parts.append(self.get_starttag_text())

    def handle_endtag(self, tag: str) -> None:
        if self.in_h1 and tag == "h1":
            self.in_h1 = False

        if self.in_date_div and tag == "div":
            self.in_date_div = False

        if not self.in_body:
            return

        if tag == "div":
            if self.body_div_depth == 1:
                self.in_body = False
                self.body_div_depth = 0
                return
            self.body_div_depth -= 1
            self.body_parts.append(f"</{tag}>")
            return

        self.body_parts.append(f"</{tag}>")

    def handle_startendtag(self, tag: str, attrs: List[tuple[str, Optional[str]]]) -> None:
        if self.in_body:
            self.body_parts.append(self.get_starttag_text())

    def handle_data(self, data: str) -> None:
        if self.in_h1:
            self.title_parts.append(data)
        if self.in_date_div:
            self.date_parts.append(data)
        if self.in_body:
            self.body_parts.append(data)

    def handle_entityref(self, name: str) -> None:
        if self.in_body:
            self.body_parts.append(f"&{name};")

    def handle_charref(self, name: str) -> None:
        if self.in_body:
            self.body_parts.append(f"&#{name};")

    def get_title(self) -> str:
        return re.sub(r"\s+", " ", "".join(self.title_parts)).strip()

    def get_date(self) -> Optional[str]:
        value = re.sub(r"\s+", " ", "".join(self.date_parts)).strip()
        if re.fullmatch(r"[A-Za-z]+\s+\d{1,2},\s+\d{4}", value):
            return value
        return None

    def get_body_html(self) -> str:
        return "".join(self.body_parts).strip()


def maybe_fix_mojibake(text: str) -> str:
    if "\u00e2" not in text:
        return text
    try:
        repaired = text.encode("cp1252", errors="ignore").decode("utf-8", errors="ignore")
    except UnicodeError:
        return text
    if repaired.count("\u00e2") < text.count("\u00e2"):
        return repaired
    return text


def normalize_common_artifacts(text: str) -> str:
    replacements = {
        "\u00e2\u20ac\u2122": "\u2019",
        "\u00e2\u20ac\u0153": "\u201c",
        "\u00e2\u20ac\x9d": "\u201d",
        "\u00e2\u20ac\u201c": "\u2013",
        "\u00e2\u20ac\u201d": "\u2014",
        "\u00a0": " ",
    }
    normalized = text
    for old, new in replacements.items():
        normalized = normalized.replace(old, new)
    return normalized


def fetch_html(url: str, timeout_seconds: float = 30.0) -> str:
    request = Request(
        url,
        headers={
            "User-Agent": "Hippodetector/1.0",
            "Accept": "text/html,application/xhtml+xml",
        },
    )
    with urlopen(request, timeout=timeout_seconds) as response:
        raw = response.read()
        charset = response.headers.get_content_charset() or "utf-8"

    decoded = raw.decode(charset, errors="replace")
    return normalize_common_artifacts(maybe_fix_mojibake(decoded))


def extract_release_links_and_next_page(html: str, page_url: str) -> tuple[List[str], Optional[str]]:
    parser = ListingParser()
    parser.feed(html)
    release_urls = sorted(urljoin(page_url, path) for path in parser.release_paths)
    next_url = urljoin(page_url, parser.next_href) if parser.next_href else None
    return release_urls, next_url


def extract_first_match(pattern: str, html: str, flags: int = 0) -> Optional[str]:
    match = re.search(pattern, html, flags)
    if not match:
        return None
    return match.group(1).strip()


def normalize_html_whitespace(html: str) -> str:
    return re.sub(r">\s+<", "><", html.strip())


def extract_release_details(html: str, url: str) -> Dict[str, Optional[str]]:
    detail_parser = DetailParser()
    detail_parser.feed(html)

    published_time = extract_first_match(
        r'<meta[^>]*property="article:published_time"[^>]*content="([^"]+)"',
        html,
        flags=re.IGNORECASE,
    )

    title_text = ""
    title_raw = detail_parser.get_title()
    if title_raw:
        title_parser = TextExtractor()
        title_parser.feed(unescape(title_raw))
        title_text = normalize_common_artifacts(maybe_fix_mojibake(title_parser.get_text()))

    body_html = detail_parser.get_body_html()
    body_html = normalize_common_artifacts(maybe_fix_mojibake(normalize_html_whitespace(body_html))) if body_html else ""

    body_text = ""
    if body_html:
        body_parser = TextExtractor()
        body_parser.feed(unescape(body_html))
        body_text = normalize_common_artifacts(maybe_fix_mojibake(body_parser.get_text()))

    return {
        "url": url,
        "title": title_text or None,
        "date": detail_parser.get_date(),
        "publishedTime": published_time,
        "bodyHtml": body_html or None,
        "bodyText": body_text or None,
    }


def parse_last_name(member_name: str) -> Optional[str]:
    if not member_name:
        return None
    last = member_name.split(",", 1)[0].strip().lower()
    clean = re.sub(r"[^a-z]", "", last)
    return clean or None


def get_terms(member: Dict[str, Any]) -> List[Dict[str, Any]]:
    terms = member.get("terms", {}).get("item", [])
    if isinstance(terms, dict):
        return [terms]
    if isinstance(terms, list):
        return [t for t in terms if isinstance(t, dict)]
    return []


def is_current_house_member(member: Dict[str, Any]) -> bool:
    for term in get_terms(member):
        if term.get("chamber") == "House of Representatives" and "endYear" not in term:
            return True
    return False


def load_member_source(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    members = payload.get("members", [])
    if not isinstance(members, list):
        raise ValueError("members file format invalid: expected top-level 'members' list")
    return [m for m in members if isinstance(m, dict)]


def scrape_member_press_releases(
    lastname: str,
    max_pages: Optional[int],
    request_delay_seconds: float,
    timeout_seconds: float,
) -> Dict[str, Any]:
    start_url = f"https://{lastname}.house.gov/media/press-releases"
    visited_pages: Set[str] = set()
    release_urls: Set[str] = set()
    page_url: Optional[str] = start_url
    pages_scraped = 0

    while page_url and page_url not in visited_pages:
        if max_pages is not None and pages_scraped >= max_pages:
            break

        visited_pages.add(page_url)
        html = fetch_html(page_url, timeout_seconds=timeout_seconds)
        page_release_urls, next_url = extract_release_links_and_next_page(html, page_url)
        release_urls.update(page_release_urls)

        pages_scraped += 1
        page_url = next_url

        if request_delay_seconds > 0:
            time.sleep(request_delay_seconds)

    releases: List[Dict[str, Optional[str]]] = []
    for release_url in sorted(release_urls):
        detail_html = fetch_html(release_url, timeout_seconds=timeout_seconds)
        releases.append(extract_release_details(detail_html, release_url))
        if request_delay_seconds > 0:
            time.sleep(request_delay_seconds)

    return {
        "source": start_url,
        "pagesScraped": pages_scraped,
        "releaseCount": len(releases),
        "pressReleases": releases,
    }


def select_members(
    all_members: List[Dict[str, Any]],
    only_current_house: bool,
    bioguide_ids: Optional[Set[str]],
    limit_members: Optional[int],
) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    for member in all_members:
        bioguide_id = member.get("bioguideId")
        if not bioguide_id:
            continue
        if only_current_house and not is_current_house_member(member):
            continue
        if bioguide_ids and bioguide_id not in bioguide_ids:
            continue
        selected.append(member)

    selected.sort(key=lambda m: (m.get("name") or "", m.get("bioguideId") or ""))
    if limit_members is not None:
        selected = selected[:limit_members]
    return selected


def scrape_members_to_bioguide_map(
    members: List[Dict[str, Any]],
    max_pages: Optional[int],
    request_delay_seconds: float,
    timeout_seconds: float,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {}

    for index, member in enumerate(members, start=1):
        bioguide_id = member.get("bioguideId")
        name = member.get("name") or ""
        lastname = parse_last_name(name)

        member_entry: Dict[str, Any] = {
            "bioguideId": bioguide_id,
            "name": name,
            "lastName": lastname,
            "state": member.get("state"),
            "partyName": member.get("partyName"),
            "status": None,
            "error": None,
            "source": None,
            "pagesScraped": 0,
            "releaseCount": 0,
            "pressReleases": [],
        }

        if not lastname:
            member_entry["status"] = "skipped"
            member_entry["error"] = "Could not parse last name from member name"
            result[bioguide_id] = member_entry
            print(f"[{index}/{len(members)}] {bioguide_id}: skipped (no parseable last name)")
            continue

        try:
            scraped = scrape_member_press_releases(
                lastname=lastname,
                max_pages=max_pages,
                request_delay_seconds=request_delay_seconds,
                timeout_seconds=timeout_seconds,
            )
            member_entry["source"] = scraped["source"]
            member_entry["pagesScraped"] = scraped["pagesScraped"]
            member_entry["releaseCount"] = scraped["releaseCount"]
            member_entry["pressReleases"] = scraped["pressReleases"]
            member_entry["status"] = "ok" if scraped["releaseCount"] > 0 else "no_releases"
            print(
                f"[{index}/{len(members)}] {bioguide_id}: "
                f"{member_entry['status']} ({member_entry['releaseCount']} releases)"
            )
        except (HTTPError, URLError, TimeoutError, ValueError) as exc:
            member_entry["status"] = "error"
            member_entry["error"] = str(exc)
            member_entry["source"] = f"https://{lastname}.house.gov/media/press-releases"
            print(f"[{index}/{len(members)}] {bioguide_id}: error ({exc})")
        except Exception as exc:
            member_entry["status"] = "error"
            member_entry["error"] = f"Unexpected error: {exc}"
            member_entry["source"] = f"https://{lastname}.house.gov/media/press-releases"
            print(f"[{index}/{len(members)}] {bioguide_id}: unexpected error ({exc})")

        result[bioguide_id] = member_entry

    return result


def summarize_status(members_by_bioguide: Dict[str, Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {"ok": 0, "no_releases": 0, "error": 0, "skipped": 0}
    for entry in members_by_bioguide.values():
        status = entry.get("status")
        if status in counts:
            counts[status] += 1
    return counts


def parse_bioguide_ids(raw: Optional[str]) -> Optional[Set[str]]:
    if not raw:
        return None
    ids = {x.strip() for x in raw.split(",") if x.strip()}
    return ids or None


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Scrape press releases from House member sites and store one combined JSON keyed by bioguideId."
        )
    )
    parser.add_argument(
        "--members-file",
        default=str(DEFAULT_MEMBERS_FILE),
        help="Path to congress_members JSON (default: data/congress_members.json)",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Output JSON path (default: data/press_releases_by_bioguide.json)",
    )
    parser.add_argument(
        "--bioguide-ids",
        default=None,
        help="Optional comma-separated bioguide IDs to scrape (subset).",
    )
    parser.add_argument(
        "--all-members",
        action="store_true",
        help="Include all members in the members file (default is current House only).",
    )
    parser.add_argument(
        "--limit-members",
        type=int,
        default=None,
        help="Optional member limit for testing.",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Optional page limit per member for testing.",
    )
    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=0.0,
        help="Optional delay between requests (default: 0).",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=30.0,
        help="HTTP timeout in seconds (default: 30).",
    )
    args = parser.parse_args()

    members_file = Path(args.members_file)
    output_path = Path(args.output)

    all_members = load_member_source(members_file)
    only_current_house = not args.all_members
    selected_members = select_members(
        all_members=all_members,
        only_current_house=only_current_house,
        bioguide_ids=parse_bioguide_ids(args.bioguide_ids),
        limit_members=args.limit_members,
    )

    print(f"Selected {len(selected_members)} members to scrape.")

    members_by_bioguide = scrape_members_to_bioguide_map(
        members=selected_members,
        max_pages=args.max_pages,
        request_delay_seconds=max(args.delay_seconds, 0.0),
        timeout_seconds=max(args.timeout_seconds, 1.0),
    )

    payload = {
        "scrapedAtUtc": datetime.now(timezone.utc).isoformat(),
        "sourceMembersFile": str(members_file),
        "selection": {
            "onlyCurrentHouse": only_current_house,
            "bioguideIds": sorted(parse_bioguide_ids(args.bioguide_ids) or []),
            "limitMembers": args.limit_members,
            "maxPagesPerMember": args.max_pages,
        },
        "selectedMemberCount": len(selected_members),
        "statusCounts": summarize_status(members_by_bioguide),
        "membersByBioguideId": members_by_bioguide,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote combined output to {output_path}")
    print(f"Status summary: {payload['statusCounts']}")


if __name__ == "__main__":
    main()
