from typing import Literal

from pydantic import BaseModel


ArchiaModelRef = Literal[
    "priv-claude-sonnet-4-5-20250929",
    "priv-claude-sonnet-4-20250514",
    "priv-claude-3-5-haiku-20241022",
    "priv-claude-3-7-sonnet-20250219",
    "priv-claude-sonnet-4-5",
    "priv-claude-haiku-4-5-20251001",
    "priv-claude-opus-4-5",
    "priv-claude-opus-4-6",
    "priv-claude-opus-4-5-20251101",
    "priv-claude-haiku-4-5",
    "priv-claude-opus-4-1-20250805",
    "gpt-5-nano",
    "gpt-5-mini",
    "gpt-5",
    "gpt-5.2",
    "gpt-5.1",
    "gpt-4.1",
    "gpt-4o",
    "gpt-4o-mini",
    "archia::openai/gpt-oss-20b",
    "archia::openai/gpt-oss-120b",
    "groq::meta-llama/llama-prompt-guard-2-86m",
    "groq::openai/gpt-oss-safeguard-20b",
    "groq::openai/gpt-oss-20b",
    "groq::openai/gpt-oss-120b",
    "grog:: meta-llama/llama-prompt-guard-2-86m",
    "archia::moonshotai/kimi-k2-instruct-0905",
]


class ArchiaModelChoice(BaseModel):
    model: ArchiaModelRef

