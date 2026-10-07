"""Optional local embedding fallback (DESIGN.md section 10).

When ``CRYPTO_LOCAL_EMBEDDING=1`` the loop app calls :func:`install`, which replaces
``rdagent.oai.llm_utils.calculate_embedding_distance_between_str_list`` (and the name already imported into
``rdagent.components.coder.CoSTEER.knowledge_management``) with a hashed unigram+bigram cosine similarity
computed in numpy. This lets the loop run without any embedding provider; the only effect is that CoSTEER's
knowledge retrieval (similar past tasks / error traces) becomes a bag-of-words match instead of a semantic
one. The LLM chat backend is untouched.
"""

from __future__ import annotations

import re
import zlib

import numpy as np

from rdagent.log import rdagent_logger as logger

N_BUCKETS = 4096
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_PATCHED_MODULES: tuple[str, ...] = (
    "rdagent.oai.llm_utils",
    "rdagent.components.coder.CoSTEER.knowledge_management",
)
_FUNC_NAME = "calculate_embedding_distance_between_str_list"


def tokenize(text: str) -> list[str]:
    """Lowercase tokens split on non-alphanumerics, plus adjacent bigrams joined with a space."""
    unigrams = _TOKEN_RE.findall(text.lower())
    bigrams = [f"{a} {b}" for a, b in zip(unigrams, unigrams[1:])]
    return unigrams + bigrams


def hashed_embedding(text: str, n_buckets: int = N_BUCKETS) -> np.ndarray:
    """Deterministic hashed bag-of-tokens vector (log1p counts, L2-normalised; zero vector for empty text)."""
    vec = np.zeros(n_buckets, dtype=np.float64)
    for tok in tokenize(text):
        vec[zlib.crc32(tok.encode("utf-8")) % n_buckets] += 1.0
    vec = np.log1p(vec)
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec /= norm
    return vec


def local_embedding_distance(source_str_list: list[str], target_str_list: list[str]) -> list[list[float]]:
    """Drop-in replacement for ``calculate_embedding_distance_between_str_list``.

    Same contract as the original: ``[[]]`` when either list is empty, otherwise a
    ``len(source) x len(target)`` nested list of cosine similarities.
    """
    if not source_str_list or not target_str_list:
        return [[]]
    src = np.vstack([hashed_embedding(s) for s in source_str_list])
    tgt = np.vstack([hashed_embedding(t) for t in target_str_list])
    return np.dot(src, tgt.T).tolist()


def install() -> list[str]:
    """Monkeypatch the embedding-distance function in every module that binds it by name.

    Idempotent. Returns the list of module paths that were patched.
    """
    import importlib

    patched: list[str] = []
    for mod_path in _PATCHED_MODULES:
        try:
            mod = importlib.import_module(mod_path)
        except Exception as e:  # noqa: BLE001 - optional module (e.g. heavy import failure)
            logger.warning(f"local_embedding: cannot import {mod_path}, not patched: {e}")
            continue
        if getattr(mod, _FUNC_NAME, None) is local_embedding_distance:
            patched.append(mod_path)
            continue
        setattr(mod, _FUNC_NAME, local_embedding_distance)
        patched.append(mod_path)
    logger.info(f"local_embedding: hashed-token cosine similarity installed in {patched}")
    return patched
