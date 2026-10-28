"""preprocessing.py - Clean bug report text so a model can use it.

Input:  DataFrame from data_loader (columns bug_id, title, description, ...)
Output: (DataFrame with a new column "clean_text", list of warning messages)
"""

import re

import pandas as pd

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
NOT_ALNUM_PATTERN = re.compile(r"[^a-z0-9]+")  # anything that is not a letter or digit

_stemmer = None  # created only when stemming is requested


def _stem_words(text):
    """Cut each word to its root, e.g. "crashing" -> "crash". Needs NLTK."""
    global _stemmer
    if _stemmer is None:
        from nltk.stem import PorterStemmer  # imported here so NLTK is optional

        _stemmer = PorterStemmer()
    return " ".join(_stemmer.stem(word) for word in text.split())


def clean_text(text, stem=False):
    """Lowercase, remove URLs and symbols, and fix spaces.

    Missing values (None, NaN) and non-text values become "".
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = URL_PATTERN.sub(" ", text)
    text = NOT_ALNUM_PATTERN.sub(" ", text)
    text = " ".join(text.split())  # removes extra spaces
    if stem and text:
        text = _stem_words(text)
    return text


def combine_title_description(title, description):
    """Join title and description into one text."""
    title = title if isinstance(title, str) else ""
    description = description if isinstance(description, str) else ""
    return (title + " " + description).strip()


def preprocess_reports(df, stem=False):
    """Add a "clean_text" column. Rows that end up with no text are removed.

    The input DataFrame is not changed; a new one is returned.
    """
    missing = [c for c in ("title", "description") if c not in df.columns]
    if missing:
        raise ValueError("DataFrame is missing column(s): " + ", ".join(missing))

    result = df.copy()
    combined = [
        combine_title_description(t, d) for t, d in zip(result["title"], result["description"])
    ]
    result["clean_text"] = [clean_text(text, stem=stem) for text in combined]

    warnings = []
    empty = result["clean_text"] == ""
    if empty.any():
        warnings.append(f"Removed {int(empty.sum())} row(s) with no usable text after cleaning.")
        result = result[~empty]

    return result.reset_index(drop=True), warnings
