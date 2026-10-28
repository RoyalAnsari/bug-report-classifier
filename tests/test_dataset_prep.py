"""Unit tests for src/dataset_prep.py (requirement FR-01, dataset adaptation)."""

import pandas as pd
import pytest

from src.dataset_prep import prepare_bug_reports


def make_raw():
    return pd.DataFrame(
        {
            "Project": ["Bugzilla", "Eclipse"],
            "Bug ID": [322082, 322082],  # same number in two projects, numeric type
            "Severity Label": ["normal", " trivial "],
            "Resolution Status": ["FIXED", "FIXED"],
            "Short Description": ["Clean up SQL ", "Typo in message"],
        }
    )


def test_columns_are_mapped():  # TC-DP-01
    result = prepare_bug_reports(make_raw())
    assert list(result.columns) == ["bug_id", "title", "description", "component"]
    assert list(result["title"]) == ["Clean up SQL", "Typo in message"]
    assert list(result["component"]) == ["normal", "trivial"]


def test_bug_id_includes_project():  # TC-DP-02
    result = prepare_bug_reports(make_raw())
    assert list(result["bug_id"]) == ["Bugzilla-322082", "Eclipse-322082"]
    assert result["bug_id"].is_unique


def test_description_is_empty():  # TC-DP-03
    assert (prepare_bug_reports(make_raw())["description"] == "").all()


def test_other_label_column():  # TC-DP-04
    result = prepare_bug_reports(make_raw(), label_column="Resolution Status")
    assert list(result["component"]) == ["FIXED", "FIXED"]


def test_missing_column_raises():  # TC-DP-05
    with pytest.raises(ValueError, match="Short Description"):
        prepare_bug_reports(make_raw().drop(columns=["Short Description"]))


def test_missing_label_column_raises():  # TC-DP-06
    with pytest.raises(ValueError, match="Nope"):
        prepare_bug_reports(make_raw(), label_column="Nope")


def test_missing_values_become_empty():  # TC-DP-07
    raw = make_raw()
    raw.loc[0, "Short Description"] = None
    raw.loc[1, "Bug ID"] = None
    result = prepare_bug_reports(raw)
    assert result.loc[0, "title"] == ""
    assert result.loc[1, "bug_id"] == ""


def test_empty_dataframe():  # TC-DP-08
    result = prepare_bug_reports(make_raw().iloc[0:0])
    assert len(result) == 0
