"""Embedding providers (spec section 22, stack section 5).

Spec section 5 requires a *configurable* embedding provider and states that the
system must not depend on one specific vendor. So this module defines the
interface, and the vector projection in ``vector.py`` knows nothing about which
implementation is behind it.

The bundled default, ``HashedTokenProvider``, is not a learned model. It is a
feature-hashing projection of tokens and character n-grams: deterministic, offline,
zero-dependency, and honest about what it is. It gives the vector projection
*morphological* similarity - ``validate_token`` retrieves ``validate_tokens`` and
survives a typo - which is more than the lexical index does and much less than
spec section 22's "conceptual matching". Nothing in MCM infers meaning from these
vectors; spec section 22 forbids the vector store from being responsible for exact
dependency reasoning, and the default provider is a standing reminder of why.

Swapping in a real model is one class and one environment variable. The provider's
``name`` is recorded against every stored vector, so changing providers invalidates
the index rather than silently mixing incomparable geometries.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
from abc import ABC, abstractmethod

#: A dense vector. Plain lists keep the core dependency-free; a NumPy-backed
#: provider is free to return something faster, as long as it indexes like this.
Vector = list[float]

_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def tokenize(text: str) -> list[str]:
    """Split text into lowercased identifier tokens.

    ``validate_token`` and ``validateToken`` both tokenise to ``[validate, token]``,
    so the two naming conventions are not treated as unrelated vocabulary. The
    joined form is kept as well, because an exact identifier match should not be
    diluted by its own parts.
    """
    out: list[str] = []
    for word in _WORD.findall(text):
        out.append(word.lower())
        parts = [p for chunk in _CAMEL.split(word) for p in chunk.split("_") if p]
        if len(parts) > 1:
            out.extend(p.lower() for p in parts)
    return out


class EmbeddingProvider(ABC):
    """Turns text into vectors of a fixed dimension."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable identity of this provider *and its configuration*.

        Stored against every vector. Two providers that would produce different
        geometries must not share a name, or the index will mix them.
        """

    @property
    @abstractmethod
    def dimension(self) -> int: ...

    @abstractmethod
    def embed(self, texts: list[str]) -> list[Vector]:
        """Embed a batch. Batching is the interface because remote providers
        charge per request, not per token."""

    def embed_one(self, text: str) -> Vector:
        return self.embed([text])[0]


class HashedTokenProvider(EmbeddingProvider):
    """Feature hashing over tokens and character n-grams. No training, no network.

    Signed hashing - the top bit of the digest decides whether a feature adds or
    subtracts - keeps collisions from systematically inflating similarity, which is
    the usual failure of naive count hashing.

    Token counts are damped with ``1 + log(count)`` so a function that repeats one
    identifier twenty times does not drown out its own name.
    """

    def __init__(self, dimension: int = 256, ngram: int = 4,
                 ngram_weight: float = 0.5) -> None:
        if dimension < 8:
            raise ValueError("dimension must be at least 8")
        self._dimension = dimension
        self._ngram = ngram
        self._ngram_weight = ngram_weight

    @property
    def name(self) -> str:
        return f"hashed-token/{self._dimension}/{self._ngram}"

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: list[str]) -> list[Vector]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> Vector:
        counts: dict[str, float] = {}
        for token in tokenize(text):
            counts[token] = counts.get(token, 0.0) + 1.0
            for gram in self._ngrams(token):
                key = "#" + gram
                counts[key] = counts.get(key, 0.0) + self._ngram_weight

        vector = [0.0] * self._dimension
        for feature, count in counts.items():
            index, sign = _hash_feature(feature, self._dimension)
            vector[index] += sign * (1.0 + math.log(count))
        return _normalise(vector)

    def _ngrams(self, token: str) -> list[str]:
        """Character n-grams inside a token, padded so prefixes and suffixes count.

        This is what lets the vector channel retrieve ``validate_tokens`` for
        ``validate_token``. The lexical index, which matches whole terms, cannot.
        """
        if len(token) < 3:
            return []
        padded = "^" + token + "$"
        n = min(self._ngram, len(padded))
        return [padded[i:i + n] for i in range(len(padded) - n + 1)]


class SentenceTransformerProvider(EmbeddingProvider):
    """Adapter for a locally-run sentence-transformers model.

    Not a dependency of this package. Import failure is reported at construction,
    not at first query, so a misconfigured provider fails before any index is built
    against it.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2",
                 device: str | None = None) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise RuntimeError(
                "sentence-transformers is not installed; "
                "pip install sentence-transformers, or use the default provider"
            ) from exc

        device = device or os.environ.get("MCM_EMBEDDING_DEVICE")
        if not device:
            try:
                import torch  # noqa: PLC0415
                if torch.cuda.is_available():
                    device = "cuda"
            except ImportError:
                pass

        self._model_name = model_name
        self._device = device
        kwargs = {}
        if device:
            kwargs["device"] = device
        self._model = SentenceTransformer(model_name, **kwargs)
        getter = getattr(self._model, "get_embedding_dimension",
                         getattr(self._model, "get_sentence_embedding_dimension", None))
        self._dimension = int(getter())

    @property
    def name(self) -> str:
        return f"sentence-transformers/{self._model_name}"

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def device(self) -> str:
        return str(getattr(self._model, "device", self._device or "cpu"))

    def embed(self, texts: list[str], batch_size: int = 32) -> list[Vector]:  # pragma: no cover
        vectors = self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            device=self._device,
        )
        if hasattr(vectors, "tolist"):
            return vectors.tolist()
        return [[float(x) for x in row] for row in vectors]


#: Providers addressable by name from configuration. A vendor HTTP provider is the
#: same three methods over an API client; none is bundled, because none can be
#: exercised offline and an untested provider in the default install would be a
#: worse lie than no provider at all.
PROVIDERS = {
    "hashed": HashedTokenProvider,
    "sentence-transformers": SentenceTransformerProvider,
}


def get_provider(spec: str | None = None) -> EmbeddingProvider:
    """Build a provider from ``name`` or ``name:argument``.

    Reads ``MCM_EMBEDDING_PROVIDER`` when no spec is given. ``hashed:512`` sets the
    dimension; ``sentence-transformers:all-mpnet-base-v2`` names a model;
    ``sentence-transformers:all-MiniLM-L6-v2@cuda`` explicitly chooses a device.
    """
    spec = spec or os.environ.get("MCM_EMBEDDING_PROVIDER") or "hashed"
    name, _, argument = spec.partition(":")
    if name not in PROVIDERS:
        known = ", ".join(sorted(PROVIDERS))
        raise ValueError(f"unknown embedding provider {name!r}; known providers: {known}")
    factory = PROVIDERS[name]
    if not argument:
        return factory()
    if factory is HashedTokenProvider:
        return factory(dimension=int(argument))
    if "@" in argument:
        model_name, _, dev = argument.partition("@")
        return factory(model_name=model_name, device=dev)
    return factory(argument)


def cosine(a: Vector, b: Vector) -> float:
    """Cosine similarity.

    Both arguments are expected to be unit vectors, so this is a dot product; it
    renormalises defensively for providers whose output is not normalised.
    """
    if len(a) != len(b):
        raise ValueError(f"dimension mismatch: {len(a)} vs {len(b)}")
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    if abs(na - 1.0) < 1e-9 and abs(nb - 1.0) < 1e-9:
        return dot
    return dot / (na * nb)


def _hash_feature(feature: str, dimension: int) -> tuple[int, float]:
    """Map a feature to a bucket and a sign.

    ``hashlib``, not the built-in ``hash()``: the built-in is randomised per
    process, and these vectors are written to disk and compared across runs.
    """
    digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, "big")
    return value % dimension, 1.0 if (value >> 63) & 1 else -1.0


def _normalise(vector: Vector) -> Vector:
    norm = math.sqrt(sum(x * x for x in vector))
    if norm == 0.0:
        return vector
    return [x / norm for x in vector]
