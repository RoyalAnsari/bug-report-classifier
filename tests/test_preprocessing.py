"""Unit tests for src/preprocessing.py (requirements FR-06, FR-07, FR-08)."""

import pandas as pd
import pytest

from src.preprocessing import clean_text, combine_title_description, preprocess_reports


# ---------- clean_text ----------
def test_lowercase():  # TC-PP-01
    assert clean_text("App CRASHES") == "app crashes"


def test_extra_spaces_removed():  # TC-PP-02
    assert clean_text("  too    many \n spaces  ") == "too many spaces"


def test_symbols_removed():  # TC-PP-03
    assert clean_text("Error!!! (code #42)") == "error code 42"


def test_url_removed():  # TC-PP-04
    assert clean_text("See https://example.com/bug?id=1 now") == "see now"


def test_none_and_nan_become_empty():  # TC-PP-05
    assert clean_text(None) == ""
    assert clean_text(float("nan")) == ""


def test_non_text_becomes_empty():  # TC-PP-06
    assert clean_text(123) == ""


def test_empty_string():  # TC-PP-07
    assert clean_text("") == ""


def test_only_symbols_becomes_empty():  # TC-PP-08
    assert clean_text("!!! ???") == ""


def test_stemming_when_nltk_available():  # TC-PP-09
    pytest.importorskip("nltk")
    assert clean_text("crashing crashes", stem=True) == "crash crash"


# ---------- combine ----------
def test_combine_title_and_description():  # TC-PP-10
    assert combine_title_description("Title", "Details") == "Title Details"


def test_combine_with_missing_description():  # TC-PP-11
    assert combine_title_description("Title", None) == "Title"


# ---------- preprocess_reports ----------
def test_adds_clean_text_column():  # TC-PP-12
    df = pd.DataFrame({"bug_id": ["1"], "title": ["App CRASH"], "description": ["On  start!"]})
    result, warnings = preprocess_reports(df)
    assert result.loc[0, "clean_text"] == "app crash on start"
    assert warnings == []


def test_input_is_not_changed():  # TC-PP-13
    df = pd.DataFrame({"bug_id": ["1"], "title": ["A"], "description": ["B"]})
    preprocess_reports(df)
    assert "clean_text" not in df.columns


def test_rows_without_text_are_removed():  # TC-PP-14
    df = pd.DataFrame({"bug_id": ["1", "2"], "title": ["Good", "???"], "description": ["bug", ""]})
    result, warnings = preprocess_reports(df)
    assert list(result["bug_id"]) == ["1"]
    assert any("Removed 1" in w for w in warnings)


def test_missing_column_raises():  # TC-PP-15
    df = pd.DataFrame({"bug_id": ["1"], "title": ["A"]})
    with pytest.raises(ValueError, match="description"):
        preprocess_reports(df)


def test_other_columns_are_kept():  # TC-PP-16
    df = pd.DataFrame(
        {"bug_id": ["1"], "title": ["A"], "description": ["B"], "component": ["UI"]}
    )
    result, _ = preprocess_reports(df)
    assert result.loc[0, "component"] == "UI"
