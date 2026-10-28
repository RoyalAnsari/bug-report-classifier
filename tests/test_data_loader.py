"""Unit tests for src/data_loader.py (requirements FR-01 to FR-04)."""

import io

import pytest

from src.data_loader import DataValidationError, load_bug_reports

HEADER = "bug_id,title,description,component\n"


def make_csv(tmp_path, text, name="reports.csv"):
    """Helper: write text to a CSV file and return its path."""
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return str(path)


# ---------- Normal cases ----------
def test_valid_file_loads(tmp_path):  # TC-DL-01
    path = make_csv(tmp_path, HEADER + "1,App crashes,Crash on start,UI\n2,Login fails,Wrong error,Auth\n")
    df, warnings = load_bug_reports(path)
    assert len(df) == 2
    assert list(df.columns) == ["bug_id", "title", "description", "component"]
    assert warnings == []


def test_file_object_works():  # TC-DL-02 (Streamlit gives file-like objects)
    df, _ = load_bug_reports(io.StringIO("bug_id,title,description\n1,A,B\n"))
    assert len(df) == 1


def test_component_column_is_optional(tmp_path):  # TC-DL-03
    path = make_csv(tmp_path, "bug_id,title,description\n1,A,B\n")
    df, _ = load_bug_reports(path)
    assert "component" not in df.columns


def test_column_names_are_normalised(tmp_path):  # TC-DL-04
    path = make_csv(tmp_path, " Bug_ID , TITLE ,Description\n1,A,B\n")
    df, _ = load_bug_reports(path)
    assert list(df.columns) == ["bug_id", "title", "description"]


def test_text_is_stripped(tmp_path):  # TC-DL-05
    path = make_csv(tmp_path, "bug_id,title,description\n1,  Hello  ,  World \n")
    df, _ = load_bug_reports(path)
    assert df.loc[0, "title"] == "Hello"
    assert df.loc[0, "description"] == "World"


def test_ids_keep_leading_zeros(tmp_path):  # TC-DL-06
    path = make_csv(tmp_path, "bug_id,title,description\n007,A,B\n")
    df, _ = load_bug_reports(path)
    assert df.loc[0, "bug_id"] == "007"


# ---------- Invalid input ----------
def test_empty_file_raises(tmp_path):  # TC-DL-07
    path = make_csv(tmp_path, "")
    with pytest.raises(DataValidationError, match="empty"):
        load_bug_reports(path)


def test_header_only_raises(tmp_path):  # TC-DL-08
    path = make_csv(tmp_path, "bug_id,title,description\n")
    with pytest.raises(DataValidationError, match="no bug reports"):
        load_bug_reports(path)


def test_missing_columns_raises(tmp_path):  # TC-DL-09
    path = make_csv(tmp_path, "bug_id,title\n1,A\n")
    with pytest.raises(DataValidationError, match="description"):
        load_bug_reports(path)


def test_missing_file_raises(tmp_path):  # TC-DL-10
    with pytest.raises(DataValidationError, match="not found"):
        load_bug_reports(str(tmp_path / "nope.csv"))


# ---------- Edge cases ----------
def test_rows_without_text_or_id_are_removed(tmp_path):  # TC-DL-11
    path = make_csv(tmp_path, "bug_id,title,description\n1,A,B\n2,,\n,C,D\n")
    df, warnings = load_bug_reports(path)
    assert list(df["bug_id"]) == ["1"]
    assert any("Removed 2" in w for w in warnings)


def test_repeated_ids_are_removed(tmp_path):  # TC-DL-12
    path = make_csv(tmp_path, "bug_id,title,description\n1,A,B\n1,C,D\n")
    df, warnings = load_bug_reports(path)
    assert len(df) == 1
    assert df.loc[0, "title"] == "A"  # first one is kept
    assert any("repeated" in w for w in warnings)


def test_empty_description_is_kept_with_warning(tmp_path):  # TC-DL-13
    path = make_csv(tmp_path, "bug_id,title,description\n1,Only title,\n")
    df, warnings = load_bug_reports(path)
    assert len(df) == 1
    assert any("empty description" in w for w in warnings)


def test_all_rows_invalid_raises(tmp_path):  # TC-DL-14
    path = make_csv(tmp_path, "bug_id,title,description\n,,\n2,,\n")
    with pytest.raises(DataValidationError, match="No valid"):
        load_bug_reports(path)
