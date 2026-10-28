"""Unit tests for src/duplicate_detector.py (requirements FR-13, FR-14, FR-15)."""

import pytest

from src.duplicate_detector import (
    RESULT_COLUMNS,
    STATUS_TEXT,
    DuplicateDetectorError,
    find_duplicate_pairs,
    find_similar_reports,
)

TEXTS = [
    "app crashes on startup",
    "application crashes on startup",
    "login fails password",
    "cannot login password fails",
    "export report slow",
]
IDS = ["a", "b", "c", "d", "e"]


def pair_set(result):
    return {frozenset((r.query_id, r.match_id)) for r in result.itertuples()}


# ---------- find_duplicate_pairs: normal cases ----------
def test_finds_similar_pairs():  # TC-DD-01
    result = find_duplicate_pairs(TEXTS, IDS, threshold=0.3)
    assert pair_set(result) == {frozenset("ab"), frozenset("cd")}


def test_result_columns_and_status():  # TC-DD-02
    result = find_duplicate_pairs(TEXTS, IDS, threshold=0.3)
    assert list(result.columns) == RESULT_COLUMNS
    assert set(result["status"]) == {STATUS_TEXT}
    assert "confirmed" not in STATUS_TEXT


def test_sorted_most_similar_first():  # TC-DD-03
    result = find_duplicate_pairs(TEXTS, IDS, threshold=0.3)
    assert list(result["similarity"]) == sorted(result["similarity"], reverse=True)


def test_identical_texts_score_one():  # TC-DD-04
    result = find_duplicate_pairs(["same words here", "same words here"], ["x", "y"], threshold=0.9)
    assert len(result) == 1
    assert result.loc[0, "similarity"] > 0.999


def test_scores_between_0_and_1():  # TC-DD-05
    result = find_duplicate_pairs(TEXTS, IDS, threshold=0.0, top_k=4)
    assert ((result["similarity"] >= 0) & (result["similarity"] <= 1)).all()


# ---------- threshold and top_k ----------
def test_high_threshold_removes_weak_pairs():  # TC-DD-06
    assert len(find_duplicate_pairs(TEXTS, IDS, threshold=0.99)) == 0


def test_lower_threshold_gives_more_or_equal_pairs():  # TC-DD-07
    low = find_duplicate_pairs(TEXTS, IDS, threshold=0.0, top_k=4)
    high = find_duplicate_pairs(TEXTS, IDS, threshold=0.7, top_k=4)
    assert len(low) >= len(high)


def test_top_k_limits_matches_per_report():  # TC-DD-08
    texts = ["same words"] * 6
    ids = list("abcdef")
    result = find_duplicate_pairs(texts, ids, threshold=0.5, top_k=1)
    assert 1 <= len(result) <= 6  # not all 15 possible pairs


# ---------- no self matches, no mirrored pairs ----------
def test_no_self_matches_and_no_mirror_pairs():  # TC-DD-09
    result = find_duplicate_pairs(["same words"] * 5, list("abcde"), threshold=0.5, top_k=4)
    assert (result["query_id"] != result["match_id"]).all()
    assert len(pair_set(result)) == len(result)  # every pair appears once
    assert len(result) == 10  # 5 reports -> 10 different pairs


# ---------- edge cases ----------
def test_empty_input_gives_empty_result():  # TC-DD-10
    result = find_duplicate_pairs([], [])
    assert len(result) == 0 and list(result.columns) == RESULT_COLUMNS


def test_single_report_gives_empty_result():  # TC-DD-11
    assert len(find_duplicate_pairs(["only one"], ["a"])) == 0


def test_empty_and_missing_texts_never_match():  # TC-DD-12
    result = find_duplicate_pairs(["", None, "real text here", "real text here"], list("abcd"), threshold=0.0)
    assert pair_set(result) == {frozenset("cd")}


def test_all_empty_texts_raise():  # TC-DD-13
    with pytest.raises(DuplicateDetectorError, match="no usable words"):
        find_duplicate_pairs(["", " "], ["a", "b"])


# ---------- invalid input ----------
def test_invalid_settings_raise():  # TC-DD-14
    with pytest.raises(DuplicateDetectorError, match="threshold"):
        find_duplicate_pairs(TEXTS, IDS, threshold=1.5)
    with pytest.raises(DuplicateDetectorError, match="threshold"):
        find_duplicate_pairs(TEXTS, IDS, threshold=-0.1)
    with pytest.raises(DuplicateDetectorError, match="top_k"):
        find_duplicate_pairs(TEXTS, IDS, top_k=0)


def test_length_mismatch_raises():  # TC-DD-15
    with pytest.raises(DuplicateDetectorError, match="same length"):
        find_duplicate_pairs(TEXTS, IDS[:3])


# ---------- find_similar_reports ----------
def test_new_report_finds_existing_match():  # TC-DD-16
    result = find_similar_reports(["application crashes at startup"], ["new1"], TEXTS, IDS, threshold=0.3)
    assert result.loc[0, "query_id"] == "new1"
    assert result.loc[0, "match_id"] in ("a", "b")
    assert result.loc[0, "status"] == STATUS_TEXT


def test_new_report_with_no_match_gives_empty():  # TC-DD-17
    result = find_similar_reports(["completely different thing"], ["n"], TEXTS, IDS, threshold=0.3)
    assert len(result) == 0


def test_new_reports_top_k_limit():  # TC-DD-18
    result = find_similar_reports(["crashes startup login password"], ["n"], TEXTS, IDS, threshold=0.0, top_k=2)
    assert len(result) == 2


def test_new_reports_empty_inputs():  # TC-DD-19
    assert len(find_similar_reports([], [], TEXTS, IDS)) == 0
    with pytest.raises(DuplicateDetectorError, match="no existing reports"):
        find_similar_reports(["x y"], ["n"], [], [])


def test_new_reports_invalid_input():  # TC-DD-20
    with pytest.raises(DuplicateDetectorError, match="same length"):
        find_similar_reports(["x"], ["n", "m"], TEXTS, IDS)
    with pytest.raises(DuplicateDetectorError, match="threshold"):
        find_similar_reports(["x"], ["n"], TEXTS, IDS, threshold=2)


def test_new_empty_text_is_skipped():  # TC-DD-21
    result = find_similar_reports(["", None], ["n1", "n2"], TEXTS, IDS, threshold=0.0)
    assert len(result) == 0


# ---------- bug found on real data (see docs/bug_log.md, BUG-01) ----------
def test_version_numbers_are_not_ignored():  # TC-DD-22
    texts = ["release notes for bugzilla 4 2 1", "release notes for bugzilla 3 4 8"]
    result = find_duplicate_pairs(texts, ["a", "b"], threshold=0.0)
    assert result.loc[0, "similarity"] < 0.9
    assert len(find_duplicate_pairs(texts, ["a", "b"], threshold=0.9)) == 0


def test_same_text_with_version_still_matches():  # TC-DD-23
    texts = ["release notes for bugzilla 4 2 1", "release notes for bugzilla 4 2 1"]
    assert len(find_duplicate_pairs(texts, ["a", "b"], threshold=0.99)) == 1
