import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse
from urllib.request import Request, urlopen


BASE_URL = "https://api.congress.gov/v3/member"
DEFAULT_LIMIT = 250
DEFAULT_OUTPUT_PATH = Path("data/congress_members.json")


def load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def ensure_api_key() -> str:
    key = (
        os.getenv("CONGRESS_API")
        or os.getenv("CONGRES_API")
        or os.getenv("CONGRESS_API_KEY")
    )
    if not key:
        raise RuntimeError(
            "Missing Congress API key. Set CONGRESS_API in your environment or .env file."
        )
    return key


def with_api_key(url: str, api_key: str) -> str:
    parsed = urlparse(url)
    params: Dict[str, str] = dict(parse_qsl(parsed.query, keep_blank_values=True))
    params["api_key"] = api_key
    updated_query = urlencode(params)
    return urlunparse(
        (
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            parsed.params,
            updated_query,
            parsed.fragment,
        )
    )


def request_json(url: str) -> Dict:
    request = Request(
        url,
        headers={
            "User-Agent": "Hippodetector/1.0",
            "Accept": "application/json",
        },
    )
    with urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_all_members(api_key: str, limit: int = DEFAULT_LIMIT) -> Tuple[List[Dict], int]:
    next_url = f"{BASE_URL}?format=json&limit={limit}"
    members: List[Dict] = []
    total_count = 0

    while next_url:
        payload = request_json(with_api_key(next_url, api_key))
        members.extend(payload.get("members", []))
        total_count = payload.get("pagination", {}).get("count", total_count)
        next_url = payload.get("pagination", {}).get("next")

    return members, total_count


def main() -> None:
    load_env_file(Path(".env"))
    api_key = ensure_api_key()

    members, expected_count = fetch_all_members(api_key=api_key)

    output_path = DEFAULT_OUTPUT_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_payload = {
        "fetchedAtUtc": datetime.now(timezone.utc).isoformat(),
        "source": BASE_URL,
        "expectedCount": expected_count,
        "fetchedCount": len(members),
        "members": members,
    }

    output_path.write_text(
        json.dumps(output_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote {len(members)} members to {output_path}")


if __name__ == "__main__":
    main()
