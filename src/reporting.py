"""reporting.py - Everything the app SHOWS or EXPORTS, as plain tested functions.

The app (app.py) only draws what these functions return. Chart functions return
small tables of numbers (not pictures), so they can be tested without a screen.
"""

import pandas as pd

from src.review import DECISIONS, REVIEW_COLUMNS, review_summary

REPORT_SORT_COLUMNS = ["bug_id", "title", "predicted_component", "confidence"]
PAIR_SORT_COLUMNS = ["similarity", "human_decision", "query_id", "match_id"]
SIMILARITY_BANDS = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]
EXPORT_REPORT_COLUMNS = ["bug_id", "title", "description", "component", "predicted_component", "confidence"]
EXPORT_PAIR_COLUMNS = [
    "query_id", "query_title", "match_id", "match_title", "similarity",
    "status", "human_decision", "reviewed_at", "final_status",
]


class ReportingError(Exception):
    """Raised when a table or setting given to a reporting function is not valid."""


def _need(table, columns):
    missing = [c for c in columns if c not in table.columns]
    if missing:
        raise ReportingError("Table is missing column(s): " + ", ".join(missing))


# ---------- overview numbers (the cards at the top of the page) ----------
def overview_metrics(reports, duplicates):
    """Numbers for the summary cards."""
    _need(reports, ["predicted_component"])
    _need(duplicates, REVIEW_COLUMNS)
    counts = review_summary(duplicates)
    top = reports["predicted_component"].value_counts()
    return {
        "reports": len(reports),
        "pairs": counts["total"],
        "confirmed": counts["confirmed"],
        "rejected": counts["rejected"],
        "pending": counts["pending"],
        "top_class": top.index[0] if len(top) else "-",
    }


# ---------- chart data ----------
def class_distribution(reports):
    """Columns: class, count, percent (biggest class first)."""
    _need(reports, ["predicted_component"])
    counts = reports["predicted_component"].value_counts()
    total = int(counts.sum())
    table = pd.DataFrame({"class": counts.index, "count": counts.values})
    table["percent"] = (table["count"] / total * 100).round(1) if total else 0.0
    return table.reset_index(drop=True)


def decision_distribution(duplicates):
    """Columns: decision, count. All three decisions are always listed."""
    _need(duplicates, ["human_decision"])
    counts = duplicates["human_decision"].value_counts()
    return pd.DataFrame({"decision": DECISIONS, "count": [int(counts.get(d, 0)) for d in DECISIONS]})


def similarity_bands(duplicates):
    """Columns: band, count. How many pairs fall into each similarity range."""
    _need(duplicates, ["similarity"])
    rows = []
    for low, high in SIMILARITY_BANDS:
        last = high == SIMILARITY_BANDS[-1][1]
        inside = (duplicates["similarity"] >= low) & (
            (duplicates["similarity"] <= high) if last else (duplicates["similarity"] < high)
        )
        rows.append({"band": f"{low:.1f}-{high:.1f}", "count": int(inside.sum())})
    return pd.DataFrame(rows)


# ---------- search, filter, sort: reports ----------
def search_reports(reports, query):
    """Keep reports whose title or description contains the text (ignores upper/lower case)."""
    _need(reports, ["title", "description"])
    query = (query or "").strip()
    if query == "":
        return reports.reset_index(drop=True)
    in_title = reports["title"].fillna("").str.contains(query, case=False, regex=False)
    in_description = reports["description"].fillna("").str.contains(query, case=False, regex=False)
    return reports[in_title | in_description].reset_index(drop=True)


def filter_reports(reports, predicted_classes=None, components=None):
    """Keep reports with the chosen predicted classes and/or the chosen true components.

    None means "do not filter on this". An empty list means "nothing selected" (no rows).
    """
    result = reports
    if predicted_classes is not None:
        _need(result, ["predicted_component"])
        result = result[result["predicted_component"].isin(predicted_classes)]
    if components is not None:
        _need(result, ["component"])
        result = result[result["component"].isin(components)]
    return result.reset_index(drop=True)


def sort_reports(reports, by="predicted_component", ascending=True):
    """Sort the reports table by one of REPORT_SORT_COLUMNS."""
    if by not in REPORT_SORT_COLUMNS:
        raise ReportingError(f"Cannot sort by '{by}'. Choose one of: {REPORT_SORT_COLUMNS}")
    _need(reports, [by])
    return reports.sort_values(by, ascending=ascending, kind="stable", na_position="last").reset_index(drop=True)


# ---------- search, filter, sort: duplicate pairs ----------
def filter_pairs(pairs, min_similarity=0.0, decisions=None, query=""):
    """Keep pairs at or above a similarity, with chosen decisions, matching some text.

    The text is searched in ids and (if present) titles of both reports.
    """
    _need(pairs, REVIEW_COLUMNS)
    if not 0 <= min_similarity <= 1:
        raise ReportingError("min_similarity must be between 0 and 1.")
    result = pairs[pairs["similarity"] >= min_similarity]
    if decisions is not None:
        bad = set(decisions) - set(DECISIONS)
        if bad:
            raise ReportingError("Unknown decision(s): " + ", ".join(sorted(bad)))
        result = result[result["human_decision"].isin(decisions)]
    query = (query or "").strip()
    if query:
        text_columns = [c for c in ["query_id", "match_id", "query_title", "match_title"] if c in result.columns]
        mask = pd.Series(False, index=result.index)
        for column in text_columns:
            mask |= result[column].astype(str).str.contains(query, case=False, regex=False)
        result = result[mask]
    return result.reset_index(drop=True)


def sort_pairs(pairs, by="similarity", ascending=False):
    """Sort the pairs table by one of PAIR_SORT_COLUMNS (most similar first by default)."""
    if by not in PAIR_SORT_COLUMNS:
        raise ReportingError(f"Cannot sort by '{by}'. Choose one of: {PAIR_SORT_COLUMNS}")
    _need(pairs, [by])
    return pairs.sort_values(by, ascending=ascending, kind="stable").reset_index(drop=True)


# ---------- CSV export ----------
def _safe_cell(value):
    """Stop spreadsheet programs from running a cell as a formula (CSV injection).
    Text that starts with = + - or @ gets a leading apostrophe."""
    if isinstance(value, str) and value[:1] in ("=", "+", "-", "@"):
        return "'" + value
    return value


def _to_safe_csv(table, columns):
    present = [c for c in columns if c in table.columns]
    safe = table[present].copy()
    for column in safe.columns:
        # Works for every column type: only text starting with = + - @ is changed.
        safe[column] = safe[column].map(_safe_cell)
    return safe.to_csv(index=False)


def reports_to_csv(reports):
    """Classification results as CSV text. clean_text is left out (it is only for the model)."""
    _need(reports, ["bug_id", "title", "predicted_component"])
    return _to_safe_csv(reports, EXPORT_REPORT_COLUMNS)


def duplicates_to_csv(pairs):
    """Duplicate suggestions as CSV text. Machine columns (status, similarity) and
    human columns (human_decision, reviewed_at, final_status) are both included, but stay separate."""
    _need(pairs, REVIEW_COLUMNS)
    return _to_safe_csv(pairs, EXPORT_PAIR_COLUMNS)
