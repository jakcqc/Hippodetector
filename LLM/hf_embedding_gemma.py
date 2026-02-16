from __future__ import annotations

import gc
import os
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

try:
    from huggingface_hub import InferenceClient
    HF_INFERENCE_AVAILABLE = True
except ImportError:
    HF_INFERENCE_AVAILABLE = False


MODEL_NAME = "google/embeddinggemma-300m"


@dataclass(frozen=True)
class NeighborResult:
    index: int
    score: float


class EmbeddingGemmaClient:
    """GPU-first embedding client with fallback: GPU → HF Inference API → CPU."""

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

        self.model_name = model_name
        self.hf_token = hf_token
        self.normalize = normalize
        self.mode = None  # Will be set to "cuda", "hf_api", or "cpu"
        self.inference_client = None

        # Determine execution mode with smart fallback
        if device is None:
            self.mode = self._select_best_mode()
        else:
            # User explicitly specified device
            if device == "hf_api":
                self.mode = "hf_api"
            else:
                self.mode = device

        # Initialize based on mode
        if self.mode == "hf_api":
            print(f"🌐 Using Hugging Face Inference API for embeddings")
            if not HF_INFERENCE_AVAILABLE:
                raise RuntimeError("huggingface_hub not installed. Run: pip install huggingface_hub")
            self.inference_client = InferenceClient(token=hf_token)
            self.device = torch.device("cpu")  # For any local tensor ops
            self.tokenizer = None
            self.model = None
        else:
            # Local execution (cuda or cpu)
            print(f"💻 Using local {self.mode.upper()} for embeddings")
            self.device = torch.device(self.mode)
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                token=hf_token,
            )
            self.model = AutoModel.from_pretrained(
                model_name,
                token=hf_token,
            ).to(self.device)
            self.model.eval()

    def _select_best_mode(self) -> str:
        """Smart fallback: GPU (if compatible) → HF API → CPU."""

        # Try GPU first
        if torch.cuda.is_available():
            try:
                capability = torch.cuda.get_device_capability(0)
                # Require compute capability >= 7.0 (sm_70)
                if capability[0] >= 7:
                    return "cuda"
                else:
                    print(f"⚠️  GPU detected (sm_{capability[0]}{capability[1]}) incompatible with PyTorch build (requires sm_70+)")
            except Exception as e:
                print(f"⚠️  Could not check GPU capability: {e}")

        # Try HF Inference API as second fallback
        if HF_INFERENCE_AVAILABLE:
            try:
                # Quick test to see if API is accessible
                test_client = InferenceClient(token=self.hf_token)
                # Don't actually call API here, just check client can be created
                print("✓ Falling back to Hugging Face Inference API (faster than CPU)")
                return "hf_api"
            except Exception as e:
                print(f"⚠️  HF Inference API not available: {e}")

        # Final fallback to CPU
        print("⚠️  Falling back to CPU (this will be slower)")
        return "cpu"

    @property
    def embedding_dim(self) -> int:
        # Full model embedding size, no truncation.
        if self.mode == "hf_api":
            return 768  # embeddinggemma-300m dimension
        return int(self.model.config.hidden_size)

    def embed_text(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def embed_texts(
        self,
        texts: Sequence[str],
        max_length: int = 8192,
        batch_size: int | None = None,
    ) -> list[list[float]]:
        if not texts:
            return []

        # Use HF Inference API if in API mode
        if self.mode == "hf_api":
            return self._embed_texts_api(texts)

        # Auto-batch on CPU to avoid OOM, process all at once on GPU
        if batch_size is None:
            batch_size = 8 if self.mode == "cpu" else len(texts)

        # Process in batches if needed
        if len(texts) <= batch_size:
            return self._embed_texts_batch(texts, max_length)

        # Batch processing with progress bar
        all_embeddings = []
        total_batches = (len(texts) + batch_size - 1) // batch_size

        with tqdm(total=len(texts), desc="Generating embeddings", unit="text") as pbar:
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_embeddings = self._embed_texts_batch(batch, max_length)
                all_embeddings.extend(batch_embeddings)
                pbar.update(len(batch))

                # Free memory between batches
                gc.collect()

        return all_embeddings

    def _embed_texts_batch(
        self,
        texts: Sequence[str],
        max_length: int = 8192,
    ) -> list[list[float]]:
        """Process a single batch of texts."""
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

    def _embed_texts_api(self, texts: Sequence[str]) -> list[list[float]]:
        """Generate embeddings using HF Inference API."""
        try:
            # API returns embeddings directly
            embeddings = []
            for text in texts:
                result = self.inference_client.feature_extraction(
                    text,
                    model=self.model_name
                )
                # Result is already a list of floats
                embeddings.append(result)

            # Normalize if requested
            if self.normalize:
                embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)
                embeddings_tensor = F.normalize(embeddings_tensor, p=2, dim=1)
                return embeddings_tensor.tolist()
            return embeddings
        except Exception as e:
            # If API fails, fall back to CPU
            print(f"⚠️  HF Inference API failed: {e}")
            print("⚠️  Falling back to CPU for this batch")
            self._fallback_to_cpu()
            return self.embed_texts(texts)  # Retry with CPU mode

    def _fallback_to_cpu(self) -> None:
        """Emergency fallback from API to CPU mode."""
        if self.mode == "hf_api":
            print("🔄 Switching to CPU mode...")
            self.mode = "cpu"
            self.device = torch.device("cpu")
            self.inference_client = None

            # Load model locally
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                token=self.hf_token,
            )
            self.model = AutoModel.from_pretrained(
                self.model_name,
                token=self.hf_token,
            ).to(self.device)
            self.model.eval()
            print("✓ Switched to CPU mode")

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
