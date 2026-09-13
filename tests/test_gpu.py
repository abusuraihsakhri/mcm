"""Tests for GPU availability, PyTorch CUDA operations, and SentenceTransformerProvider."""

from __future__ import annotations

import math
import pytest

try:
    import torch
except ImportError:
    torch = None

try:
    import sentence_transformers
except ImportError:
    sentence_transformers = None

from mcm.retrieval.embedding import (
    SentenceTransformerProvider,
    get_provider,
    cosine,
)


@pytest.mark.skipif(torch is None, reason="PyTorch is not installed")
class TestGPUBasics:
    def test_torch_cuda_available(self):
        assert torch is not None
        assert torch.cuda.is_available(), "CUDA is not available in PyTorch"
        assert torch.cuda.device_count() >= 1
        device_name = torch.cuda.get_device_name(0)
        assert len(device_name) > 0

    def test_cuda_tensor_matmult(self):
        device = torch.device("cuda:0")
        a = torch.randn(128, 64, device=device)
        b = torch.randn(64, 32, device=device)
        c = torch.matmul(a, b)
        assert c.shape == (128, 32)
        assert c.is_cuda


@pytest.fixture(scope="module")
def gpu_provider():
    device = "cuda" if (torch and torch.cuda.is_available()) else "cpu"
    return SentenceTransformerProvider(model_name="all-MiniLM-L6-v2", device=device)


@pytest.mark.skipif(sentence_transformers is None, reason="sentence-transformers not installed")
class TestSentenceTransformerGPU:

    def test_provider_initialization(self, gpu_provider):
        assert gpu_provider.dimension == 384
        assert "all-MiniLM-L6-v2" in gpu_provider.name
        if torch and torch.cuda.is_available():
            assert "cuda" in gpu_provider.device.lower()

    def test_embed_and_embed_one_parity(self, gpu_provider):
        text = "def authenticate(token: str) -> bool:"
        vec1 = gpu_provider.embed_one(text)
        vecs = gpu_provider.embed([text])
        assert len(vecs) == 1
        vec2 = vecs[0]
        assert len(vec1) == gpu_provider.dimension
        assert len(vec2) == gpu_provider.dimension

        norm1 = math.sqrt(sum(x * x for x in vec1))
        norm2 = math.sqrt(sum(x * x for x in vec2))
        assert abs(norm1 - 1.0) < 1e-4
        assert abs(norm2 - 1.0) < 1e-4

        assert cosine(vec1, vec2) > 0.9999

    def test_batch_embedding(self, gpu_provider):
        texts = [
            "def validate_token(token): pass",
            "class AuthenticationError(Exception): pass",
            "import jwt",
            "def test_login_flow(): pass",
        ]
        vectors = gpu_provider.embed(texts)
        assert len(vectors) == 4
        for v in vectors:
            assert len(v) == gpu_provider.dimension
            norm = math.sqrt(sum(x * x for x in v))
            assert abs(norm - 1.0) < 1e-4

        sim_auth = cosine(vectors[0], vectors[1])
        assert isinstance(sim_auth, float)

    def test_get_provider_with_device_spec(self):
        provider = get_provider("sentence-transformers:all-MiniLM-L6-v2@cpu")
        assert isinstance(provider, SentenceTransformerProvider)
        assert "cpu" in provider.device.lower()
