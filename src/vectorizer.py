"""
vectorizer.py
-------------
The article's "Text Vectorizer" stage uses Doc2Vec: an unsupervised model
trained offline on a large labeled corpus of mortgage pages to produce a
fixed-length embedding per page.

We don't have that corpus here (it's proprietary, per the article itself),
so this module swaps in TF-IDF as a practical, dependency-light stand-in
that needs zero training data and fits on the fly, per package. The
interface is intentionally the same shape as a Doc2Vec wrapper would be
(`fit`, `transform`) so a real trained Doc2Vec/embedding model can be
dropped in later without touching the rest of the pipeline -- see
`DocEmbeddingVectorizer` docstring for the swap point.

This also matches the "efficiency" constraint in the InfrX brief: TF-IDF
on cleaned page text is essentially free at 2,000-page scale, where a
heavyweight embedding model for every single page would not be.
"""
from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .text_clean import clean_text


class DocEmbeddingVectorizer:
    """Fixed-length vector representation for page text.

    Swap point: replace the TfidfVectorizer below with a trained Doc2Vec
    model (`gensim.models.Doc2Vec`) and implement `.infer_vector(tokens)`
    in `transform` -- the rest of the pipeline only depends on `fit` and
    `transform` returning an (n_pages, n_features) array, so nothing else
    needs to change.
    """

    def __init__(self, max_features: int = 4096):
        self._vec = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        self._fitted = False

    def fit(self, page_texts: list[str]) -> "DocEmbeddingVectorizer":
        cleaned = [clean_text(t) for t in page_texts]
        # Guard against an all-empty package (e.g. every page failed OCR).
        if not any(cleaned):
            cleaned = ["empty"] * len(cleaned)
        self._vec.fit(cleaned)
        self._fitted = True
        return self

    def transform(self, page_texts: list[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")
        cleaned = [clean_text(t) or "empty" for t in page_texts]
        return self._vec.transform(cleaned).toarray()

    def fit_transform(self, page_texts: list[str]) -> np.ndarray:
        self.fit(page_texts)
        return self.transform(page_texts)


def consecutive_similarities(vectors: np.ndarray) -> np.ndarray:
    """Cosine similarity between each page and the page right before it.

    A sharp drop in this value is the unsupervised signal that "something
    changed" between page i-1 and page i -- used as a fallback boundary
    cue when no rule-based document signature fires.
    """
    n = vectors.shape[0]
    sims = np.ones(n)
    for i in range(1, n):
        sims[i] = cosine_similarity(vectors[i : i + 1], vectors[i - 1 : i])[0, 0]
    return sims
