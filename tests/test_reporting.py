"""Unit tests for src/reporting.py (requirements FR-17, FR-18, FR-19)."""

import io

import pandas as pd
import pytest

from src.review import create_review_table, set_decision
from src.reporting import (
    ReportingError,
    class_distribution,
    decision_distribution,
    duplicates_to_csv,
    filter_pairs,
    filter_reports,
    overview_metrics,
    reports_to_csv,
    search_reports,
    similarity_bands,
    sort_pairs,
    sort_reports,
)


def make_reports():
    return pd.DataFrame(
        {
            "bug_id": ["1", "2", "3", "4"],
            "title": ["App CRASH on save", "Typo in label", "Crash again", "Slow export"],
            "description": ["", "see dialog", "on startup", ""],
            "component": ["major", "trivial", "major", "normal"],
            "clean_text": ["a", "b", "c", "d"],
            "predicted_component": ["major", "trivial", "major", "normal"],
            "confidence": [None, None, None, None],
        }
    )


def make_pairs():
    pairs = pd.DataFrame(
        {
            "query_id": ["1", "2", "3"],
            "match_id": ["3", "4", "4"],
            "similarity": [0.95, 0.72, 0.55],
            "status": ["potential duplicate"] * 3,
        }
    )
    table = create_review_table(pairs)
    table.insert(2, "query_title", ["App CRASH on save", "Typo in label", "Crash again"])
    table.insert(4, "match_title", ["Crash again", "Slow export", "Slow export"])
    return table


# ---------- overview ----------
def test_overview_metrics():  # TC-RP-01
    pairs = set_decision(make_pairs(), "1", "3", "confirmed")
    metrics = overview_metrics(make_reports(), pairs)
    assert metrics == {"reports": 4, "pairs": 3, "confirmed": 1, "rejected": 0, "pending": 2, "top_class": "major"}


def test_overview_with_no_pairs():  # TC-RP-02
    empty = create_review_table(pd.DataFrame(columns=["query_id", "match_id", "similarity", "status"]))
    assert overview_metrics(make_reports(), empty)["pairs"] == 0


# ---------- chart data ----------
def test_class_distribution():  # TC-RP-03
    table = class_distribution(make_reports())
    assert list(table.columns) == ["class", "count", "percent"]
    assert table.iloc[0]["class"] == "major" and table.iloc[0]["count"] == 2
    assert table["percent"].sum() == 100.0


def test_class_distribution_empty():  # TC-RP-04
    assert len(class_distribution(make_reports().iloc[0:0])) == 0


def test_decision_distribution_lists_all_three():  # TC-RP-05
    table = decision_distribution(set_decision(make_pairs(), "1", "3", "rejected"))
    assert list(table["decision"]) == ["pending", "confirmed", "rejected"]
    assert list(table["count"]) == [2, 0, 1]


def test_similarity_bands():  # TC-RP-06
    bands = similarity_bands(make_pairs())
    counts = dict(zip(bands["band"], bands["count"]))
    assert counts["0.9-1.0"] == 1 and counts["0.7-0.8"] == 1 and counts["0.5-0.6"] == 1
    assert bands["count"].sum() == 3


def test_similarity_band_edges():  # TC-RP-07 (1.0 counts in the last band; 0.8 in the 0.8-0.9 band)
    pairs = make_pairs()
    pairs["similarity"] = [1.0, 0.8, 0.5]
    counts = dict(zip(*[similarity_bands(pairs)[c] for c in ("band", "count")]))
    assert counts["0.9-1.0"] == 1 and counts["0.8-0.9"] == 1 and counts["0.5-0.6"] == 1


# ---------- search / filter / sort reports ----------
def test_search_title_and_description():  # TC-RP-08
    assert list(search_reports(make_reports(), "crash")["bug_id"]) == ["1", "3"]
    assert list(search_reports(make_reports(), "DIALOG")["bug_id"]) == ["2"]


def test_search_empty_returns_all_and_no_match_returns_none():  # TC-RP-09
    assert len(search_reports(make_reports(), "")) == 4
    assert len(search_reports(make_reports(), None)) == 4
    assert len(search_reports(make_reports(), "zzz")) == 0


def test_search_special_characters_do_not_crash():  # TC-RP-10
    assert len(search_reports(make_reports(), "(*[")) == 0


def test_filter_by_predicted_class():  # TC-RP-11
    assert list(filter_reports(make_reports(), predicted_classes=["major"])["bug_id"]) == ["1", "3"]
    assert len(filter_reports(make_reports(), predicted_classes=[])) == 0
    assert len(filter_reports(make_reports())) == 4


def test_filter_by_component():  # TC-RP-12
    assert list(filter_reports(make_reports(), components=["normal"])["bug_id"]) == ["4"]


def test_filter_missing_column_raises():  # TC-RP-13
    with pytest.raises(ReportingError, match="component"):
        filter_reports(make_reports().drop(columns=["component"]), components=["x"])


def test_sort_reports():  # TC-RP-14
    assert list(sort_reports(make_reports(), "title")["bug_id"]) == ["1", "3", "4", "2"]
    assert list(sort_reports(make_reports(), "bug_id", ascending=False)["bug_id"]) == ["4", "3", "2", "1"]
    with pytest.raises(ReportingError, match="Cannot sort"):
        sort_reports(make_reports(), "nonsense")


# ---------- filter / sort pairs ----------
def test_filter_pairs_by_similarity():  # TC-RP-15
    assert len(filter_pairs(make_pairs(), min_similarity=0.7)) == 2
    assert len(filter_pairs(make_pairs(), min_similarity=1.0)) == 0
    with pytest.raises(ReportingError, match="between 0 and 1"):
        filter_pairs(make_pairs(), min_similarity=2)


def test_filter_pairs_by_decision():  # TC-RP-16
    pairs = set_decision(make_pairs(), "1", "3", "confirmed")
    assert len(filter_pairs(pairs, decisions=["confirmed"])) == 1
    assert len(filter_pairs(pairs, decisions=["pending", "rejected"])) == 2
    with pytest.raises(ReportingError, match="Unknown decision"):
        filter_pairs(pairs, decisions=["maybe"])


def test_filter_pairs_by_text():  # TC-RP-17
    assert len(filter_pairs(make_pairs(), query="slow")) == 2
    assert len(filter_pairs(make_pairs(), query="  ")) == 3


def test_sort_pairs():  # TC-RP-18
    assert list(sort_pairs(make_pairs())["similarity"]) == [0.95, 0.72, 0.55]
    assert list(sort_pairs(make_pairs(), ascending=True)["similarity"]) == [0.55, 0.72, 0.95]
    with pytest.raises(ReportingError, match="Cannot sort"):
        sort_pairs(make_pairs(), "nonsense")


# ---------- CSV export ----------
def test_reports_csv_columns():  # TC-RP-19
    text = reports_to_csv(make_reports())
    header = text.splitlines()[0]
    assert header == "bug_id,title,description,component,predicted_component,confidence"
    assert "clean_text" not in text
    assert len(pd.read_csv(io.StringIO(text))) == 4


def test_duplicates_csv_keeps_machine_and_human_columns():  # TC-RP-20
    pairs = set_decision(make_pairs(), "1", "3", "confirmed", now="2026-10-28T10:00:00+00:00")
    table = pd.read_csv(io.StringIO(duplicates_to_csv(pairs)), dtype=str)
    assert list(table.columns) == [
        "query_id", "query_title", "match_id", "match_title", "similarity",
        "status", "human_decision", "reviewed_at", "final_status",
    ]
    assert table.loc[0, "status"] == "potential duplicate"
    assert table.loc[0, "human_decision"] == "confirmed"


def test_duplicates_csv_without_titles_still_works():  # TC-RP-21
    pairs = make_pairs().drop(columns=["query_title", "match_title"])
    assert "query_title" not in duplicates_to_csv(pairs).splitlines()[0]


def test_csv_injection_is_blocked():  # TC-RP-22
    reports = make_reports()
    reports.loc[0, "title"] = "=HYPERLINK(\"http://x\")"
    reports.loc[1, "title"] = "@SUM(A1)"
    reports.loc[2, "title"] = "+1+1"
    text = reports_to_csv(reports)
    table = pd.read_csv(io.StringIO(text), dtype=str)
    assert table.loc[0, "title"].startswith("'=")
    assert table.loc[1, "title"].startswith("'@")
    assert table.loc[2, "title"].startswith("'+")
    assert table.loc[3, "title"] == "Slow export"  # normal text unchanged


def test_export_empty_tables():  # TC-RP-23
    assert reports_to_csv(make_reports().iloc[0:0]).startswith("bug_id,title")
    empty = make_pairs().iloc[0:0]
    assert duplicates_to_csv(empty).startswith("query_id")


def test_export_missing_columns_raises():  # TC-RP-24
    with pytest.raises(ReportingError, match="predicted_component"):
        reports_to_csv(pd.DataFrame({"bug_id": ["1"], "title": ["a"]}))
