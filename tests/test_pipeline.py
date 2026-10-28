"""Unit tests for src/pipeline.py (the end-to-end backend workflow)."""

import io
import json

import pytest

from src.classifier import save_model, train_classifier
from src.pipeline import PipelineError, _model_uses_stemming, add_titles, run_pipeline
from src.review import REVIEW_COLUMNS

WORDS = {
    "normal": "update docs page link",
    "major": "crash exception fails broken",
    "trivial": "typo spelling label",
}


def make_model(tmp_path):
    """Train a small model and save it. Returns the model path."""
    texts, labels = [], []
    for label, words in WORDS.items():
        for i in range(15):
            texts.append(f"{words} item{i}")
            labels.append(label)
    path = str(tmp_path / "models" / "classifier.joblib")
    save_model(train_classifier(texts, labels), path)
    return path


CSV = (
    "bug_id,title,description,component\n"
    "1,App crash exception on save,,major\n"
    "2,Crash exception when saving,,major\n"
    "3,Typo in label,,trivial\n"
    "4,Update docs page,,normal\n"
)


# ---------- normal run ----------
def test_full_run(tmp_path):  # TC-PL-01
    result = run_pipeline(io.StringIO(CSV), make_model(tmp_path))
    assert set(result) == {"reports", "duplicates", "warnings", "summary"}
    reports = result["reports"]
    assert len(reports) == 4
    for col in ["clean_text", "predicted_component", "confidence"]:
        assert col in reports.columns
    assert result["summary"]["reports"] == 4


def test_predictions_make_sense(tmp_path):  # TC-PL-02
    reports = run_pipeline(io.StringIO(CSV), make_model(tmp_path))["reports"]
    assert reports.loc[reports["bug_id"] == "1", "predicted_component"].iloc[0] == "major"
    assert reports.loc[reports["bug_id"] == "3", "predicted_component"].iloc[0] == "trivial"


def test_duplicates_are_a_pending_review_table(tmp_path):  # TC-PL-03
    dup = run_pipeline(io.StringIO(CSV), make_model(tmp_path), threshold=0.3)["duplicates"]
    assert list(dup.columns) == REVIEW_COLUMNS
    assert len(dup) >= 1
    assert (dup["human_decision"] == "pending").all()
    assert set(dup["status"]) == {"potential duplicate"}


def test_threshold_changes_pair_count(tmp_path):  # TC-PL-04
    model = make_model(tmp_path)
    low = run_pipeline(io.StringIO(CSV), model, threshold=0.0)["summary"]["duplicate_pairs"]
    high = run_pipeline(io.StringIO(CSV), model, threshold=0.99)["summary"]["duplicate_pairs"]
    assert low >= high


def test_summary_counts_match(tmp_path):  # TC-PL-05
    result = run_pipeline(io.StringIO(CSV), make_model(tmp_path), threshold=0.3)
    assert sum(result["summary"]["predicted_classes"].values()) == 4
    assert result["summary"]["duplicate_pairs"] == len(result["duplicates"])


# ---------- label column and unseen categories ----------
def test_works_without_component_column(tmp_path):  # TC-PL-06
    csv = "bug_id,title,description\n1,crash exception,\n2,typo label,\n"
    reports = run_pipeline(io.StringIO(csv), make_model(tmp_path))["reports"]
    assert "component" not in reports.columns and len(reports) == 2


def test_unseen_category_in_file_does_not_crash(tmp_path):  # TC-PL-07
    csv = "bug_id,title,description,component\n1,crash exception,,never_seen_before\n2,typo label,,\n"
    reports = run_pipeline(io.StringIO(csv), make_model(tmp_path))["reports"]
    assert len(reports) == 2
    assert set(reports["predicted_component"]) <= set(WORDS)  # predictions use known classes only


# ---------- warnings from earlier modules reach the user ----------
def test_warnings_are_collected(tmp_path):  # TC-PL-08
    csv = "bug_id,title,description\n1,crash exception,\n1,repeat id,x\n2,???,\n"
    warnings = run_pipeline(io.StringIO(csv), make_model(tmp_path))["warnings"]
    assert any("repeated" in w for w in warnings)
    assert any("empty description" in w for w in warnings)
    assert any("no usable text" in w for w in warnings)


# ---------- errors become PipelineError ----------
def test_missing_model_raises(tmp_path):  # TC-PL-09
    with pytest.raises(PipelineError, match="train_model"):
        run_pipeline(io.StringIO(CSV), str(tmp_path / "none.joblib"))


def test_missing_columns_raises(tmp_path):  # TC-PL-10
    with pytest.raises(PipelineError, match="Missing required"):
        run_pipeline(io.StringIO("bug_id,title\n1,a\n"), make_model(tmp_path))


def test_empty_file_raises(tmp_path):  # TC-PL-11
    with pytest.raises(PipelineError, match="empty"):
        run_pipeline(io.StringIO(""), make_model(tmp_path))


def test_missing_file_raises(tmp_path):  # TC-PL-12
    with pytest.raises(PipelineError, match="not found"):
        run_pipeline(str(tmp_path / "nofile.csv"), make_model(tmp_path))


def test_too_many_reports_raises(tmp_path):  # TC-PL-13
    model = make_model(tmp_path)
    with pytest.raises(PipelineError, match="limit"):
        run_pipeline(io.StringIO(CSV), model, max_reports=3)
    assert len(run_pipeline(io.StringIO(CSV), model, max_reports=None)["reports"]) == 4


def test_invalid_threshold_raises(tmp_path):  # TC-PL-14
    with pytest.raises(PipelineError, match="threshold"):
        run_pipeline(io.StringIO(CSV), make_model(tmp_path), threshold=2)


def test_all_text_unusable_raises(tmp_path):  # TC-PL-15
    csv = "bug_id,title,description\n1,???,\n2,!!!,\n"
    with pytest.raises(PipelineError, match="No usable"):
        run_pipeline(io.StringIO(csv), make_model(tmp_path))


# ---------- stemming setting follows the saved model ----------
def test_model_uses_stemming_helper(tmp_path):  # TC-PL-16
    model_path = str(tmp_path / "classifier.joblib")
    assert _model_uses_stemming(model_path) is False  # no scores file
    (tmp_path / "test_scores.json").write_text(json.dumps({"stemming": True}), encoding="utf-8")
    assert _model_uses_stemming(model_path) is True
    (tmp_path / "test_scores.json").write_text(json.dumps({"stemming": False}), encoding="utf-8")
    assert _model_uses_stemming(model_path) is False
    (tmp_path / "test_scores.json").write_text("not json", encoding="utf-8")
    assert _model_uses_stemming(model_path) is False


# ---------- add_titles ----------
def test_add_titles(tmp_path):  # TC-PL-17
    result = run_pipeline(io.StringIO(CSV), make_model(tmp_path), threshold=0.3)
    shown = add_titles(result["duplicates"], result["reports"])
    first = shown.iloc[0]
    titles = dict(zip(result["reports"]["bug_id"], result["reports"]["title"]))
    assert first["query_title"] == titles[first["query_id"]]
    assert first["match_title"] == titles[first["match_id"]]
    assert "human_decision" in shown.columns


def test_add_titles_on_empty_table(tmp_path):  # TC-PL-18
    result = run_pipeline(io.StringIO(CSV), make_model(tmp_path), threshold=0.99)
    shown = add_titles(result["duplicates"], result["reports"])
    assert len(shown) == 0 and "query_title" in shown.columns
