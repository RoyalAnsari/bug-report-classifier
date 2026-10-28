"""duplicate_detector.py - Suggest POSSIBLE duplicate bug reports.

Method (baseline): TF-IDF turns each cleaned text into numbers; "cosine similarity"
compares two texts. 1.0 = same words, 0.0 = no words in common.

Important: matches are only SUGGESTIONS ("potential duplicate"). A human decides.

Two functions:
    find_duplicate_pairs  - look for duplicates INSIDE one list of reports
    find_similar_reports  - look for existing reports similar to NEW reports
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

STATUS_TEXT = "potential duplicate"  # never "confirmed": a human must confirm
RESULT_COLUMNS = ["query_id", "match_id", "similarity", "status"]
DEFAULT_THRESHOLD = 0.8
DEFAULT_TOP_K = 5


class DuplicateDetectorError(Exception):
    """Raised when the input to the duplicate detector is not valid."""


def _check_settings(threshold, top_k):
    if not 0 <= threshold <= 1:
        raise DuplicateDetectorError("threshold must be between 0 and 1.")
    if not isinstance(top_k, int) or top_k < 1:
        raise DuplicateDetectorError("top_k must be a whole number of at least 1.")


def _clean_list(texts):
    return [t if isinstance(t, str) else "" for t in texts]


def _fit_vectorizer(reference_texts):
    """Learn the vocabulary from the reference texts. Fails if there are no real words."""
    # token_pattern keeps ONE-character words such as the digits in "4 2 1".
    # The default setting drops them, so "release 4.2.1" and "release 3.0" looked identical.
    vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
    try:
        matrix = vectorizer.fit_transform(reference_texts)
    except ValueError:
        raise DuplicateDetectorError("The reports contain no usable words.")
    return vectorizer, matrix


def _nearest(query_matrix, reference_matrix, count):
    """For every query row, return (similarities, indexes) of its nearest reference rows.

    NearestNeighbors works in small chunks, so it also works for a large dataset.
    """
    count = min(count, reference_matrix.shape[0])
    finder = NearestNeighbors(n_neighbors=count, metric="cosine", algorithm="brute")
    finder.fit(reference_matrix)
    distances, indexes = finder.kneighbors(query_matrix)
    similarities = (1 - distances).clip(0, 1)  # clip fixes tiny rounding errors
    return similarities, indexes


def find_duplicate_pairs(texts, ids, threshold=DEFAULT_THRESHOLD, top_k=DEFAULT_TOP_K):
    """Find possible duplicates inside one list of reports.

    Each pair is returned once. Returns a DataFrame with the columns
    query_id, match_id, similarity, status, sorted with the most similar first.
    """
    texts, ids = _clean_list(texts), list(ids)
    _check_settings(threshold, top_k)
    if len(texts) != len(ids):
        raise DuplicateDetectorError("texts and ids must have the same length.")
    if len(texts) < 2:
        return pd.DataFrame(columns=RESULT_COLUMNS)

    _, matrix = _fit_vectorizer(texts)
    # Ask for top_k + 1 because a report is always most similar to itself.
    similarities, indexes = _nearest(matrix, matrix, top_k + 1)

    best = {}  # (smaller index, larger index) -> similarity
    for row in range(len(texts)):
        if texts[row].strip() == "":
            continue
        kept = 0
        for sim, col in zip(similarities[row], indexes[row]):
            if col == row:
                continue
            if kept == top_k:
                break
            kept += 1
            if sim >= threshold and texts[col].strip() != "":
                key = (min(row, col), max(row, col))
                best[key] = max(best.get(key, 0), sim)

    rows = [
        {"query_id": ids[a], "match_id": ids[b], "similarity": float(s), "status": STATUS_TEXT}
        for (a, b), s in best.items()
    ]
    return _to_result(rows)


def find_similar_reports(new_texts, new_ids, reference_texts, reference_ids,
                         threshold=DEFAULT_THRESHOLD, top_k=DEFAULT_TOP_K):
    """For each NEW report, find the most similar EXISTING reports.

    Returns a DataFrame with the columns query_id (the new report), match_id
    (the existing report), similarity, status.
    """
    new_texts, new_ids = _clean_list(new_texts), list(new_ids)
    reference_texts, reference_ids = _clean_list(reference_texts), list(reference_ids)
    _check_settings(threshold, top_k)
    if len(new_texts) != len(new_ids) or len(reference_texts) != len(reference_ids):
        raise DuplicateDetectorError("texts and ids must have the same length.")
    if len(reference_texts) == 0:
        raise DuplicateDetectorError("There are no existing reports to compare with.")
    if len(new_texts) == 0:
        return pd.DataFrame(columns=RESULT_COLUMNS)

    vectorizer, reference_matrix = _fit_vectorizer(reference_texts)
    query_matrix = vectorizer.transform(new_texts)
    similarities, indexes = _nearest(query_matrix, reference_matrix, top_k)

    rows = []
    for row in range(len(new_texts)):
        if new_texts[row].strip() == "":
            continue
        for sim, col in zip(similarities[row], indexes[row]):
            if sim >= threshold and reference_texts[col].strip() != "":
                rows.append({"query_id": new_ids[row], "match_id": reference_ids[col],
                             "similarity": float(sim), "status": STATUS_TEXT})
    return _to_result(rows)


def _to_result(rows):
    if not rows:
        return pd.DataFrame(columns=RESULT_COLUMNS)
    result = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    return result.sort_values("similarity", ascending=False).reset_index(drop=True)
