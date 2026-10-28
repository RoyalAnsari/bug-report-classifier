"""Unit tests for src/review.py (requirement FR-15, FR-16)."""

import pandas as pd
import pytest

from src.review import (
    DECISIONS,
    REVIEW_COLUMNS,
    ReviewError,
    create_review_table,
    filter_by_decision,
    load_review,
    merge_previous_decisions,
    review_summary,
    save_review,
    set_decision,
)

NOW = "2026-10-28T12:00:00+00:00"


def make_pairs():
    return pd.DataFrame(
        {
            "query_id": ["a", "c", "e"],
            "match_id": ["b", "d", "f"],
            "similarity": [0.95, 0.85, 0.80],
            "status": ["potential duplicate"] * 3,
        }
    )


# ---------- create_review_table ----------
def test_create_table_starts_pending():  # TC-RV-01
    table = create_review_table(make_pairs())
    assert list(table.columns) == REVIEW_COLUMNS
    assert (table["human_decision"] == "pending").all()
    assert (table["reviewed_at"] == "").all()
    assert (table["final_status"] == "potential duplicate (not reviewed)").all()


def test_create_table_missing_column_raises():  # TC-RV-02
    with pytest.raises(ReviewError, match="similarity"):
        create_review_table(make_pairs().drop(columns=["similarity"]))


def test_create_table_from_empty():  # TC-RV-03
    empty = pd.DataFrame(columns=["query_id", "match_id", "similarity", "status"])
    table = create_review_table(empty)
    assert len(table) == 0 and list(table.columns) == REVIEW_COLUMNS


def test_create_does_not_change_input():  # TC-RV-04
    pairs = make_pairs()
    create_review_table(pairs)
    assert "human_decision" not in pairs.columns


# ---------- set_decision ----------
def test_confirm_a_pair():  # TC-RV-05
    table = set_decision(create_review_table(make_pairs()), "a", "b", "confirmed", now=NOW)
    row = table.iloc[0]
    assert row["human_decision"] == "confirmed"
    assert row["reviewed_at"] == NOW
    assert row["final_status"] == "confirmed duplicate (by human)"


def test_reject_a_pair():  # TC-RV-06
    table = set_decision(create_review_table(make_pairs()), "c", "d", "rejected", now=NOW)
    assert table.iloc[1]["human_decision"] == "rejected"
    assert table.iloc[1]["final_status"] == "not a duplicate (rejected by human)"


def test_machine_columns_are_never_changed():  # TC-RV-07
    before = create_review_table(make_pairs())
    after = set_decision(before, "a", "b", "confirmed")
    assert list(after["status"]) == list(before["status"]) == ["potential duplicate"] * 3
    assert list(after["similarity"]) == list(before["similarity"])


def test_other_pairs_stay_pending():  # TC-RV-08
    table = set_decision(create_review_table(make_pairs()), "a", "b", "confirmed")
    assert list(table["human_decision"]) == ["confirmed", "pending", "pending"]


def test_pair_found_in_either_order():  # TC-RV-09
    table = set_decision(create_review_table(make_pairs()), "b", "a", "confirmed")
    assert table.iloc[0]["human_decision"] == "confirmed"


def test_undo_with_pending():  # TC-RV-10
    table = set_decision(create_review_table(make_pairs()), "a", "b", "confirmed", now=NOW)
    table = set_decision(table, "a", "b", "pending")
    assert table.iloc[0]["human_decision"] == "pending"
    assert table.iloc[0]["reviewed_at"] == ""


def test_change_mind():  # TC-RV-11
    table = set_decision(create_review_table(make_pairs()), "a", "b", "confirmed")
    table = set_decision(table, "a", "b", "rejected")
    assert table.iloc[0]["human_decision"] == "rejected"


def test_set_decision_does_not_change_input():  # TC-RV-12
    table = create_review_table(make_pairs())
    set_decision(table, "a", "b", "confirmed")
    assert (table["human_decision"] == "pending").all()


def test_unknown_pair_raises():  # TC-RV-13
    with pytest.raises(ReviewError, match="not found"):
        set_decision(create_review_table(make_pairs()), "a", "zzz", "confirmed")


def test_invalid_decision_raises():  # TC-RV-14
    with pytest.raises(ReviewError, match="Decision must be"):
        set_decision(create_review_table(make_pairs()), "a", "b", "maybe")


def test_ids_given_as_numbers_work():  # TC-RV-15
    pairs = make_pairs()
    pairs["query_id"], pairs["match_id"] = ["1", "3", "5"], ["2", "4", "6"]
    table = set_decision(create_review_table(pairs), 1, 2, "confirmed")
    assert table.iloc[0]["human_decision"] == "confirmed"


# ---------- summary / filter ----------
def test_summary_counts():  # TC-RV-16
    table = create_review_table(make_pairs())
    table = set_decision(table, "a", "b", "confirmed")
    table = set_decision(table, "c", "d", "rejected")
    assert review_summary(table) == {"pending": 1, "confirmed": 1, "rejected": 1, "total": 3}


def test_summary_of_empty_table():  # TC-RV-17
    empty = create_review_table(pd.DataFrame(columns=["query_id", "match_id", "similarity", "status"]))
    assert review_summary(empty) == {"pending": 0, "confirmed": 0, "rejected": 0, "total": 0}


def test_filter_by_decision():  # TC-RV-18
    table = set_decision(create_review_table(make_pairs()), "a", "b", "confirmed")
    assert len(filter_by_decision(table, "confirmed")) == 1
    assert len(filter_by_decision(table, "pending")) == 2
    with pytest.raises(ReviewError):
        filter_by_decision(table, "maybe")


# ---------- merge_previous_decisions ----------
def test_merge_keeps_old_decisions():  # TC-RV-19
    old = set_decision(create_review_table(make_pairs()), "a", "b", "confirmed", now=NOW)
    new = create_review_table(make_pairs().iloc[[0, 2]])  # a new table without pair c-d
    merged = merge_previous_decisions(new, old)
    assert list(merged["human_decision"]) == ["confirmed", "pending"]
    assert merged.iloc[0]["reviewed_at"] == NOW


def test_merge_ignores_pending_and_missing_pairs():  # TC-RV-20
    old = create_review_table(make_pairs())
    new = create_review_table(make_pairs())
    assert (merge_previous_decisions(new, old)["human_decision"] == "pending").all()


# ---------- save / load ----------
def test_save_and_load_round_trip(tmp_path):  # TC-RV-21
    table = set_decision(create_review_table(make_pairs()), "a", "b", "confirmed", now=NOW)
    path = str(tmp_path / "reviews" / "decisions.csv")
    save_review(table, path)
    loaded = load_review(path)
    assert list(loaded["human_decision"]) == list(table["human_decision"])
    assert list(loaded["similarity"]) == [0.95, 0.85, 0.80]
    assert loaded.iloc[0]["reviewed_at"] == NOW


def test_load_missing_file_raises(tmp_path):  # TC-RV-22
    with pytest.raises(ReviewError, match="not found"):
        load_review(str(tmp_path / "none.csv"))


def test_load_empty_file_raises(tmp_path):  # TC-RV-23
    path = tmp_path / "empty.csv"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ReviewError, match="empty"):
        load_review(str(path))


def test_load_bad_decision_raises(tmp_path):  # TC-RV-24
    table = create_review_table(make_pairs())
    table.loc[0, "human_decision"] = "maybe"
    path = str(tmp_path / "bad.csv")
    table.to_csv(path, index=False)
    with pytest.raises(ReviewError, match="Unknown decision"):
        load_review(path)


def test_load_missing_column_raises(tmp_path):  # TC-RV-25
    path = tmp_path / "bad.csv"
    path.write_text("query_id,match_id\na,b\n", encoding="utf-8")
    with pytest.raises(ReviewError, match="missing column"):
        load_review(str(path))


def test_decisions_constant():  # TC-RV-26
    assert DECISIONS == ["pending", "confirmed", "rejected"]


# ---------- decisions_to_store (keeps decisions from other files) ----------
from src.review import decisions_to_store


def test_store_only_human_decisions():  # TC-RV-27
    table = set_decision(create_review_table(make_pairs()), "a", "b", "confirmed", now=NOW)
    stored = decisions_to_store(table)
    assert len(stored) == 1 and stored.iloc[0]["human_decision"] == "confirmed"


def test_store_with_no_decisions_is_empty():  # TC-RV-28
    assert len(decisions_to_store(create_review_table(make_pairs()))) == 0


def test_store_keeps_decisions_from_other_files():  # TC-RV-29
    old_pairs = pd.DataFrame(
        {"query_id": ["x"], "match_id": ["y"], "similarity": [0.9], "status": ["potential duplicate"]}
    )
    saved = set_decision(create_review_table(old_pairs), "x", "y", "rejected", now=NOW)
    current = set_decision(create_review_table(make_pairs()), "a", "b", "confirmed", now=NOW)
    stored = decisions_to_store(current, saved)
    assert set(stored["query_id"]) == {"a", "x"}


def test_store_current_decision_wins_over_saved():  # TC-RV-30
    saved = set_decision(create_review_table(make_pairs()), "a", "b", "confirmed", now=NOW)
    current = set_decision(create_review_table(make_pairs()), "a", "b", "rejected", now=NOW)
    stored = decisions_to_store(current, saved)
    assert len(stored) == 1 and stored.iloc[0]["human_decision"] == "rejected"


def test_store_undo_removes_decision_even_with_saved_copy():  # TC-RV-31
    saved = set_decision(create_review_table(make_pairs()), "a", "b", "confirmed", now=NOW)
    current = set_decision(saved, "a", "b", "pending")  # the person undid it
    assert len(decisions_to_store(current, saved)) == 0


def test_store_invalid_table_raises():  # TC-RV-32
    with pytest.raises(ReviewError, match="missing column"):
        decisions_to_store(pd.DataFrame({"a": [1]}))
