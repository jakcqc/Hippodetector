import json
import os
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv
from openai import OpenAI


DEFAULT_BASE_URL = "https://api.archgw.com/v1"
DEFAULT_MODELS: Dict[str, str] = {
    "openai": "openai/gpt-5-mini",
    "anthropic": "anthropic/claude-3-5-haiku-20241022",
    "google": "google/gemini-2.5-flash",
}


def _read_env() -> None:
    env_path = Path(".env")
    if env_path.exists():
        load_dotenv(env_path, override=False)
    else:
        load_dotenv(override=False)


def _build_auth_header(raw_secret: str) -> str:
    token = raw_secret.strip()
    if token.lower().startswith("bearer ") or token.lower().startswith("basic "):
        return token
    return f"Bearer {token}"


def _response_text(response: Any) -> str:
    text = getattr(response, "output_text", None)
    if text:
        return text.strip()
    return json.dumps(response.model_dump(), ensure_ascii=False)


def _provider_models() -> Dict[str, str]:
    return {
        "openai": os.getenv("ARCHIA_MODEL_OPENAI", DEFAULT_MODELS["openai"]),
        "anthropic": os.getenv(
            "ARCHIA_MODEL_ANTHROPIC", DEFAULT_MODELS["anthropic"]
        ),
        "google": os.getenv("ARCHIA_MODEL_GOOGLE", DEFAULT_MODELS["google"]),
    }


def main() -> None:
    _read_env()

    archia_key = os.getenv("ARCHIA")
    if not archia_key:
        raise RuntimeError("Missing ARCHIA in .env or environment.")

    base_url = os.getenv("ARCHIA_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
    auth_header = _build_auth_header(archia_key)

    client = OpenAI(
        api_key="archia-key-via-header",
        base_url=base_url,
        default_headers={"Authorization": auth_header},
    )

    models = _provider_models()
    print(f"Archia gateway: {base_url}")
    print()

    for provider, model in models.items():
        response = client.responses.create(
            model=model,
            input=f"Hello world from Archia using {provider}.",
        )
        print(f"[{provider}] {model}")
        print(_response_text(response))
        print()


if __name__ == "__main__":
    main()
