import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv
import httpx


DEFAULT_BASE_URL = "https://api.archia.app/v1"
TARGET_PROVIDERS = ("openai", "anthropic", "google")
DEFAULT_MODELS: Dict[str, str] = {
    "openai": "gpt-5-mini",
    "anthropic": "priv-claude-3-5-haiku-20241022",
    "google": "gemini-2.5-flash",
}


def _read_env() -> None:
    env_path = Path(".env")
    if env_path.exists():
        load_dotenv(env_path, override=False)
    else:
        load_dotenv(override=False)


def _response_text(response: Any) -> str:
    if isinstance(response, dict):
        output = response.get("output", [])
        if output and isinstance(output, list):
            content = output[0].get("content", [])
            if content and isinstance(content, list):
                text = content[0].get("text")
                if isinstance(text, str) and text.strip():
                    return text.strip()
    return json.dumps(response, ensure_ascii=False)


def _provider_models() -> Dict[str, str]:
    return {
        "openai": os.getenv("ARCHIA_MODEL_OPENAI", DEFAULT_MODELS["openai"]),
        "anthropic": os.getenv(
            "ARCHIA_MODEL_ANTHROPIC", DEFAULT_MODELS["anthropic"]
        ),
        "google": os.getenv("ARCHIA_MODEL_GOOGLE", DEFAULT_MODELS["google"]),
    }


def _fetch_archia_models(client: httpx.Client, base_url: str) -> list[dict]:
    response = client.get(f"{base_url}/models")
    response.raise_for_status()
    payload = response.json()
    return payload.get("models", [])


def _resolve_provider_models(client: httpx.Client, base_url: str) -> Dict[str, str]:
    catalog = _fetch_archia_models(client, base_url)
    preferred = _provider_models()
    selected: Dict[str, str] = {}

    for provider in TARGET_PROVIDERS:
        requested = preferred[provider]
        exact = next(
            (
                m
                for m in catalog
                if m.get("provider") == provider and m.get("system_name") == requested
            ),
            None,
        )
        if exact:
            selected[provider] = requested
            continue

        fallback = next((m for m in catalog if m.get("provider") == provider), None)
        if fallback:
            selected[provider] = str(fallback.get("system_name"))

    return selected


def _print_archia_model_list(client: httpx.Client, base_url: str) -> None:
    catalog = _fetch_archia_models(client, base_url)
    print("Archia model list (provider -> system_name):")
    for item in catalog:
        provider = item.get("provider", "unknown")
        system_name = item.get("system_name", "unknown")
        print(f"- {provider} -> {system_name}")
    print()


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    _read_env()

    archia_key = os.getenv("ARCHIA")
    if not archia_key:
        raise RuntimeError("Missing ARCHIA in .env or environment.")

    base_url = os.getenv("ARCHIA_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
    headers = {
        "x-api-key": archia_key.strip(),
        "Authorization": f"Bearer {archia_key.strip()}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }

    with httpx.Client(http2=True, timeout=60, headers=headers) as client:
        _print_archia_model_list(client=client, base_url=base_url)
        models = _resolve_provider_models(client=client, base_url=base_url)
        print(f"Archia gateway: {base_url}")
        print()

        for provider in TARGET_PROVIDERS:
            model = models.get(provider)
            if not model:
                print(f"[{provider}] no model available in your Archia account; skipping.")
                print()
                continue

            payload = {
                "model": model,
                "input": f"Hello world from Archia using {provider}.",
            }
            response = client.post(f"{base_url}/responses", json=payload)
            response.raise_for_status()
            parsed = response.json()
            message_text = _response_text(parsed)
            safe_text = message_text.encode("utf-8", errors="replace").decode(
                "utf-8", errors="replace"
            )

            print(f"[{provider}] {model}")
            print(safe_text)
            print()


if __name__ == "__main__":
    main()
