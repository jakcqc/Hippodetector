from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from transformers import AutoModel, AutoTokenizer


MODEL_NAME = "google/embeddinggemma-300m"


@dataclass(frozen=True)
class NeighborResult:
    index: int
    score: float


class EmbeddingGemmaClient:
    """GPU-first embedding client for google/embeddinggemma-300m."""

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        token_env: str = "HUGGING_FACE_API",
        device: str | None = None,
        normalize: bool = True,
    ) -> None:
        load_dotenv(override=False)
        hf_token = os.getenv(token_env)
        if not hf_token:
            raise RuntimeError(f"Missing {token_env} in environment or .env.")

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.normalize = normalize

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=hf_token,
        )
        self.model = AutoModel.from_pretrained(
            model_name,
            token=hf_token,
        ).to(self.device)
        self.model.eval()

    @property
    def embedding_dim(self) -> int:
        # Full model embedding size, no truncation.
        return int(self.model.config.hidden_size)

    def embed_text(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def embed_texts(
        self,
        texts: Sequence[str],
        max_length: int = 8192,
    ) -> list[list[float]]:
        if not texts:
            return []

        inputs = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = self._mean_pool(
                outputs.last_hidden_state,
                inputs["attention_mask"],
            )
            if self.normalize:
                embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings.detach().cpu().tolist()

    @staticmethod
    def _mean_pool(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        summed = (token_embeddings * mask).sum(dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts

    def nearest_neighbors(
        self,
        search_vector: Sequence[float],
        candidate_vectors: Sequence[Sequence[float]],
        k: int = 5,
    ) -> list[NeighborResult]:
        if not candidate_vectors:
            return []

        query = torch.tensor(search_vector, dtype=torch.float32, device=self.device)
        candidates = torch.tensor(candidate_vectors, dtype=torch.float32, device=self.device)

        if query.dim() != 1:
            raise ValueError("search_vector must be 1D.")
        if candidates.dim() != 2:
            raise ValueError("candidate_vectors must be a 2D array-like.")
        if query.shape[0] != candidates.shape[1]:
            raise ValueError("search_vector dimension must match candidate vector size.")

        query = F.normalize(query, p=2, dim=0)
        candidates = F.normalize(candidates, p=2, dim=1)
        similarities = torch.mv(candidates, query)

        k = max(1, min(k, candidates.shape[0]))
        scores, indices = torch.topk(similarities, k=k, largest=True, sorted=True)
        return [
            NeighborResult(index=int(i.item()), score=float(s.item()))
            for i, s in zip(indices, scores)
        ]

    def nearest_neighbors_from_texts(
        self,
        search_text: str,
        candidate_texts: Sequence[str],
        k: int = 5,
    ) -> list[NeighborResult]:
        if not candidate_texts:
            return []
        query_vec = self.embed_text(search_text)
        candidate_vecs = self.embed_texts(candidate_texts)
        return self.nearest_neighbors(query_vec, candidate_vecs, k=k)


def build_default_client() -> EmbeddingGemmaClient:
    """Starter helper that loads model/tokenizer once and reuses GPU when available."""
    return EmbeddingGemmaClient()


if __name__ == "__main__":
    client = build_default_client()
    sample_texts = [
        "Horse racing regulations in California",
        "Federal transportation funding for highways",
        "How to train a machine learning classifier",
    ]
    query = "state-level horse race policy"
    results = client.nearest_neighbors_from_texts(query, sample_texts, k=2)
    print(f"Embedding dimension: {client.embedding_dim}")
    print("Top neighbors:")
    for item in results:
        print(f"index={item.index} score={item.score:.4f} text={sample_texts[item.index]}")
