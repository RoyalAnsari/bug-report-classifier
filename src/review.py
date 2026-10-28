"""review.py - Human review of suggested duplicate pairs.

The machine only SUGGESTS. A person decides. This module keeps the two apart:
    status          - the machine's suggestion ("potential duplicate"); never changed here
    human_decision  - "pending", "confirmed" or "rejected"; only a person sets this
    final_status    - plain-text summary that combines the two

Input:  the table from duplicate_detector (query_id, match_id, similarity, status)
Output: the same table plus human_decision, reviewed_at, final_status
"""

import os
from datetime import datetime, timezone

import pandas as pd

REQUIRED_COLUMNS = ["query_id", "match_id", "similarity", "status"]
REVIEW_COLUMNS = REQUIRED_COLUMNS + ["human_decision", "reviewed_at", "final_status"]
DECISIONS = ["pending", "confirmed", "rejected"]
FINAL_STATUS = {
    "pending": "potential duplicate (not reviewed)",
    "confirmed": "confirmed duplicate (by human)",
    "rejected": "not a duplicate (rejected by human)",
}


class ReviewError(Exception):
    """Raised for invalid review tables, unknown pairs, or bad decisions."""


def _check_columns(table, columns):
    missing = [c for c in columns if c not in table.columns]
    if missing:
        raise ReviewError("Table is missing column(s): " + ", ".join(missing))


def _pair_key(id_a, id_b):
    """The same pair in either order gives the same key."""
    return tuple(sorted((str(id_a), str(id_b))))


def create_review_table(pairs):
    """Turn duplicate_detector output into a review table. All decisions start as 'pending'."""
    _check_columns(pairs, REQUIRED_COLUMNS)
    table = pairs[REQUIRED_COLUMNS].copy().reset_index(drop=True)
    table["human_decision"] = "pending"
    table["reviewed_at"] = ""
    table["final_status"] = FINAL_STATUS["pending"]
    return table[REVIEW_COLUMNS]


def set_decision(table, query_id, match_id, decision, now=None):
    """Record a person's decision for one pair. Returns a new table (the old one is not changed).

    decision: "confirmed", "rejected", or "pending" (to undo an earlier decision).
    The pair is found in either order. The machine columns are never touched.
    """
    _check_columns(table, REVIEW_COLUMNS)
    if decision not in DECISIONS:
        raise ReviewError(f"Decision must be one of {DECISIONS}, not '{decision}'.")

    keys = [_pair_key(q, m) for q, m in zip(table["query_id"], table["match_id"])]
    wanted = _pair_key(query_id, match_id)
    rows = [i for i, key in enumerate(keys) if key == wanted]
    if not rows:
        raise ReviewError(f"Pair not found: {query_id} and {match_id}.")

    result = table.copy()
    stamp = "" if decision == "pending" else (now or datetime.now(timezone.utc).isoformat(timespec="seconds"))
    for i in rows:
        result.iloc[i, result.columns.get_loc("human_decision")] = decision
        result.iloc[i, result.columns.get_loc("reviewed_at")] = stamp
        result.iloc[i, result.columns.get_loc("final_status")] = FINAL_STATUS[decision]
    return result


def review_summary(table):
    """Count the pairs by decision: {'total', 'pending', 'confirmed', 'rejected'}."""
    _check_columns(table, REVIEW_COLUMNS)
    counts = table["human_decision"].value_counts()
    summary = {name: int(counts.get(name, 0)) for name in DECISIONS}
    summary["total"] = len(table)
    return summary


def filter_by_decision(table, decision):
    """Return only the rows with the given decision."""
    _check_columns(table, REVIEW_COLUMNS)
    if decision not in DECISIONS:
        raise ReviewError(f"Decision must be one of {DECISIONS}, not '{decision}'.")
    return table[table["human_decision"] == decision].reset_index(drop=True)


def merge_previous_decisions(new_table, old_table):
    """Keep earlier human decisions when the suggestions are made again.

    Example: the user moves the threshold slider and gets a new table.
    Pairs that were already confirmed or rejected keep their decision.
    """
    _check_columns(new_table, REVIEW_COLUMNS)
    _check_columns(old_table, REVIEW_COLUMNS)
    old = {
        _pair_key(q, m): (d, t)
        for q, m, d, t in zip(old_table["query_id"], old_table["match_id"],
                              old_table["human_decision"], old_table["reviewed_at"])
        if d != "pending"
    }
    result = new_table.copy()
    for i, (q, m) in enumerate(zip(result["query_id"], result["match_id"])):
        key = _pair_key(q, m)
        if key in old:
            decision, stamp = old[key]
            result.iloc[i, result.columns.get_loc("human_decision")] = decision
            result.iloc[i, result.columns.get_loc("reviewed_at")] = stamp
            result.iloc[i, result.columns.get_loc("final_status")] = FINAL_STATUS[decision]
    return result


def save_review(table, path):
    """Save the review table as CSV (creates the folder if needed)."""
    _check_columns(table, REVIEW_COLUMNS)
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    table.to_csv(path, index=False)


def load_review(path):
    """Load a review table saved with save_review."""
    try:
        table = pd.read_csv(path, dtype=str, keep_default_na=False)
    except FileNotFoundError:
        raise ReviewError(f"Review file not found: {path}")
    except pd.errors.EmptyDataError:
        raise ReviewError("The review file is empty.")
    _check_columns(table, REVIEW_COLUMNS)
    bad = set(table["human_decision"]) - set(DECISIONS)
    if bad:
        raise ReviewError("Unknown decision value(s) in file: " + ", ".join(sorted(bad)))
    table["similarity"] = table["similarity"].astype(float)
    return table[REVIEW_COLUMNS]


def decisions_to_store(current, saved=None):
    """Rows to keep on disk: every human decision (confirmed or rejected) from the
    current table, plus earlier decisions for pairs that are NOT in the current table.

    This way, reviewing file B does not erase decisions made while reviewing file A.
    """
    _check_columns(current, REVIEW_COLUMNS)
    decided_now = current[current["human_decision"] != "pending"][REVIEW_COLUMNS]
    if saved is None or len(saved) == 0:
        return decided_now.reset_index(drop=True)

    _check_columns(saved, REVIEW_COLUMNS)
    current_keys = {_pair_key(q, m) for q, m in zip(current["query_id"], current["match_id"])}
    keep = [
        i for i, (q, m, d) in enumerate(zip(saved["query_id"], saved["match_id"], saved["human_decision"]))
        if d != "pending" and _pair_key(q, m) not in current_keys
    ]
    older = saved.iloc[keep][REVIEW_COLUMNS]
    return pd.concat([decided_now, older], ignore_index=True)
