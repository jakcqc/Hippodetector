import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv
import httpx
from pydantic import ValidationError

try:
    from llm.archia_model_types import ArchiaModelChoice, ArchiaModelRef
except ModuleNotFoundError:
    from archia_model_types import ArchiaModelChoice, ArchiaModelRef


DEFAULT_BASE_URL = "https://api.archia.app/v1"
TARGET_PROVIDERS = ("openai", "anthropic", "google")
DEFAULT_OPENAI_MODEL: ArchiaModelRef = "gpt-5-mini"
DEFAULT_ANTHROPIC_MODEL: ArchiaModelRef = "priv-claude-3-5-haiku-20241022"


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


def _to_model_ref(value: str) -> Optional[ArchiaModelRef]:
    try:
        choice = ArchiaModelChoice(model=value)
        return choice.model
    except ValidationError:
        return None


def _provider_models() -> Dict[str, Optional[ArchiaModelRef]]:
    openai_requested = os.getenv("ARCHIA_MODEL_OPENAI", DEFAULT_OPENAI_MODEL)
    anthropic_requested = os.getenv("ARCHIA_MODEL_ANTHROPIC", DEFAULT_ANTHROPIC_MODEL)
    google_requested = os.getenv("ARCHIA_MODEL_GOOGLE", "")
    return {
        "openai": _to_model_ref(openai_requested),
        "anthropic": _to_model_ref(anthropic_requested),
        "google": _to_model_ref(google_requested) if google_requested else None,
    }


def _fetch_archia_models(client: httpx.Client, base_url: str) -> list[dict]:
    response = client.get(f"{base_url}/models")
    response.raise_for_status()
    payload = response.json()
    return payload.get("models", [])


def _resolve_provider_models(
    client: httpx.Client, base_url: str
) -> Dict[str, ArchiaModelRef]:
    catalog = _fetch_archia_models(client, base_url)
    preferred = _provider_models()
    selected: Dict[str, ArchiaModelRef] = {}

    for provider in TARGET_PROVIDERS:
        requested = preferred[provider]
        if not requested:
            continue
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
            system_name = str(fallback.get("system_name"))
            fallback_model = _to_model_ref(system_name)
            if fallback_model:
                selected[provider] = fallback_model

    return selected


def _print_archia_model_list(client: httpx.Client, base_url: str) -> None:
    print("Fetching full model catalog...\n")
    catalog = _fetch_archia_models(client, base_url)

    print("All models (provider -> system_name | type | capabilities):")
    for item in catalog:
        provider = item.get("provider", "unknown")
        system_name = item.get("system_name", "unknown")
        model_type = item.get("type", "unknown")
        capabilities = item.get("capabilities", [])
        print(f"- {provider} -> {system_name} | type={model_type} | capabilities={capabilities}")
    print()

    # Try detecting embedding models from catalog
    embedding_models = [
        m for m in catalog
        if (
            str(m.get("type", "")).lower() == "embedding"
            or "embedding" in str(m.get("capabilities", "")).lower()
        )
    ]

    if embedding_models:
        print("Embedding-capable models found in catalog:")
        for m in embedding_models:
            print(f"- {m.get('provider')} -> {m.get('system_name')}")
    else:
        print("No embedding models detected in base catalog.\n")

    # Try explicit embedding query
    print("Trying explicit embedding model query...\n")
    try:
        response = client.get(f"{base_url}/models?type=embedding")
        response.raise_for_status()
        payload = response.json()
        models = payload.get("models", [])

        if models:
            print("Embedding models via ?type=embedding:")
            for m in models:
                print(f"- {m.get('provider')} -> {m.get('system_name')}")
        else:
            print("No models returned for ?type=embedding.")
    except Exception as e:
        print(f"Embedding query failed: {e}")

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

            typed_model: ArchiaModelRef = model

            payload = {
                "model": typed_model,
                "input": f"Hello world from Archia using {provider}.",
            }
            response = client.post(f"{base_url}/responses", json=payload)
            response.raise_for_status()
            parsed = response.json()
            message_text = _response_text(parsed)
            safe_text = message_text.encode("utf-8", errors="replace").decode(
                "utf-8", errors="replace"
            )

            print(f"[{provider}] {typed_model}")
            print(safe_text)
            print()


if __name__ == "__main__":
    main()
