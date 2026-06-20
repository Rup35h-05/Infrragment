"""
page_classifier.py
-------------------
Implements the article's "Machine Learning Classes" idea: every page is
classified as one of

    FIRST   - opens a document (a "DocType" class)
    LAST    - closes a multi-page document (a "DocType-last" class)
    OTHER   - a middle page of whatever document is currently open

Two interchangeable classifiers are provided:

  HeuristicClassifier
      No labeled data required. Combines the generic "Page X of Y" footer
      cue with the per-type title signatures in doc_signatures.py, and
      falls back to embedding-similarity change-point detection
      (consecutive_similarities) for document types with no fixed
      template (e.g. free-text borrower letters). This is the default --
      it is what makes the project runnable on day one, on an unfamiliar
      file, with no training set.

  TrainableClassifier
      A direct re-implementation of the article's training pipeline:
      DocEmbeddingVectorizer (TF-IDF stand-in for Doc2Vec) + multinomial
      LogisticRegression over (DocType, DocType-last, Other) labels,
      trained from a CSV of (page_text, ml_class) rows exactly like the
      article's "Page Text" / "ML Class" columns. Use this once you have
      real labeled samples; it will typically out-perform the heuristic
      path the same way the article reports (~93-94% accuracy) because it
      learns document-specific vocabulary instead of relying on a fixed
      regex list.

Both return the same output shape so the rest of the pipeline (boundary
detection) doesn't care which one produced it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression

from . import doc_signatures as sig
from .vectorizer import DocEmbeddingVectorizer, consecutive_similarities

SIMILARITY_BOUNDARY_THRESHOLD = 0.12  # below this -> treat as a new document


@dataclass
class PagePrediction:
    page_number: int
    role: str  # "FIRST" | "OTHER" | "LAST"
    doc_type: str
    confidence: float


class HeuristicClassifier:
    """Zero-training-data classifier (see module docstring)."""

    def classify(self, page_texts: list[str]) -> list[PagePrediction]:
        n = len(page_texts)
        vec = DocEmbeddingVectorizer().fit(page_texts)
        vectors = vec.transform(page_texts)
        sims = consecutive_similarities(vectors)

        preds: list[PagePrediction] = []
        open_type: str | None = None

        for i, text in enumerate(page_texts):
            page_no = i + 1
            page_of = sig.PAGE_OF_PATTERN.search(text)
            title_match = sig.match_doc_type(text)

            if page_of:
                cur, total = int(page_of.group(1)), int(page_of.group(2))
                if cur == 1:
                    doc_type = title_match or "Unclassified"
                    preds.append(PagePrediction(page_no, "FIRST", doc_type, 0.9))
                    open_type = doc_type
                elif cur >= total:
                    doc_type = open_type or title_match or "Unclassified"
                    preds.append(PagePrediction(page_no, "LAST", doc_type, 0.9))
                    open_type = None
                else:
                    doc_type = open_type or title_match or "Unclassified"
                    preds.append(PagePrediction(page_no, "OTHER", doc_type, 0.7))
                continue

            if title_match:
                # A title signature fired with no "Page X of Y" footer ->
                # treat as the first (and, by default, only) page of a
                # document of this type.
                preds.append(PagePrediction(page_no, "FIRST", title_match, 0.75))
                open_type = title_match
                continue

            # No structural or title cue available (free-text page, or a
            # continuation page whose template doesn't repeat the title).
            # Fall back to the unsupervised similarity-drop signal.
            if i == 0 or sims[i] < SIMILARITY_BOUNDARY_THRESHOLD or open_type is None:
                doc_type = "Unclassified"
                preds.append(PagePrediction(page_no, "FIRST", doc_type, 0.4))
                open_type = doc_type
            else:
                preds.append(PagePrediction(page_no, "OTHER", open_type, 0.5))

        return preds


class TrainableClassifier:
    """Supervised TF-IDF + Logistic Regression classifier (article re-implementation).

    Train with `fit(page_texts, ml_classes)` where `ml_classes` are labels
    like "Form_1040", "Form_1040-last", "Other" -- exactly the
    DocumentIdentifierID / "-last" / "Other" scheme from the article.
    """

    def __init__(self):
        self.vectorizer = DocEmbeddingVectorizer()
        self.clf = LogisticRegression(max_iter=1000)
        self._fitted = False

    def fit(self, page_texts: list[str], ml_classes: list[str]) -> "TrainableClassifier":
        X = self.vectorizer.fit_transform(page_texts)
        self.clf.fit(X, ml_classes)
        self._fitted = True
        return self

    def classify(self, page_texts: list[str]) -> list[PagePrediction]:
        if not self._fitted:
            raise RuntimeError("Call fit() before classify(), or load a saved model.")
        X = self.vectorizer.transform(page_texts)
        probs = self.clf.predict_proba(X)
        classes = self.clf.classes_
        preds = []
        for i, row in enumerate(probs):
            best_idx = int(np.argmax(row))
            label = classes[best_idx]
            confidence = float(row[best_idx])
            if label == "Other":
                role, doc_type = "OTHER", "Other"
            elif label.endswith("-last"):
                role, doc_type = "LAST", label[: -len("-last")]
            else:
                role, doc_type = "FIRST", label
            preds.append(PagePrediction(i + 1, role, doc_type, confidence))
        return preds

    def save(self, path: str) -> None:
        import joblib

        joblib.dump({"vectorizer": self.vectorizer, "clf": self.clf}, path)

    @classmethod
    def load(cls, path: str) -> "TrainableClassifier":
        import joblib

        obj = cls()
        data = joblib.load(path)
        obj.vectorizer = data["vectorizer"]
        obj.clf = data["clf"]
        obj._fitted = True
        return obj
