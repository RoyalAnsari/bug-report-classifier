"""Unit tests for src/train_model.py (requirements FR-09, FR-10, FR-12, FR-20)."""

import json
import os

import pandas as pd
import pytest

from src.classifier import load_model, predict
from src.train_model import prepare_training_data, run_training

WORDS = {"normal": "update docs page link", "major": "crash exception fails broken", "trivial": "typo spelling label"}


def write_csv(path, per_class=40):
    """Imbalanced fake data: 'normal' has 3x more rows than the others."""
    lines = ["bug_id,title,description,component"]
    n = 0
    for label, words in WORDS.items():
        for i in range(per_class * (3 if label == "normal" else 1)):
            n += 1
            lines.append(f"{n},{words} item{i},,{label}")
    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)


# ---------- prepare_training_data ----------
def test_prepare_removes_unlabelled_and_repeated():  # TC-TR-01
    df = pd.DataFrame({"clean_text": ["a b", "a b", "c d", "e f"], "component": ["x", "x", "y", ""]})
    texts, labels, info = prepare_training_data(df)
    assert texts == ["a b", "c d"] and labels == ["x", "y"]
    assert info == {"rows_without_label_removed": 1, "repeated_texts_removed": 1, "rows_used": 2}


def test_prepare_missing_column_raises():  # TC-TR-02
    with pytest.raises(ValueError, match="component"):
        prepare_training_data(pd.DataFrame({"clean_text": ["a"]}))


def test_prepare_missing_label_values():  # TC-TR-03
    df = pd.DataFrame({"clean_text": ["a", "b"], "component": [None, "x"]})
    _, labels, info = prepare_training_data(df)
    assert labels == ["x"] and info["rows_without_label_removed"] == 1


# ---------- run_training ----------
def test_run_training_creates_files(tmp_path):  # TC-TR-04
    csv_path = write_csv(tmp_path / "reports.csv")
    model_dir = str(tmp_path / "models")
    report = run_training(csv_path, model_dir)
    for name in ["classifier.joblib", "model_comparison.csv", "test_confusion_matrix.csv", "test_scores.json"]:
        assert os.path.exists(os.path.join(model_dir, name)), name
    assert report["best_model"] in ["logistic_regression", "naive_bayes", "linear_svm"]
    assert sum(report["split_sizes"].values()) == report["data_info"]["rows_used"]


def test_saved_model_can_predict(tmp_path):  # TC-TR-05
    csv_path = write_csv(tmp_path / "reports.csv")
    report = run_training(csv_path, str(tmp_path / "models"))
    result = predict(load_model(report["model_path"]), ["crash exception broken"])
    assert result.loc[0, "predicted_component"] == "major"


def test_scores_file_content(tmp_path):  # TC-TR-06
    csv_path = write_csv(tmp_path / "reports.csv")
    model_dir = str(tmp_path / "models")
    run_training(csv_path, model_dir)
    with open(os.path.join(model_dir, "test_scores.json"), encoding="utf-8") as f:
        scores = json.load(f)
    assert 0 <= scores["f1_macro"] <= 1
    assert 0 < scores["majority_baseline_accuracy"] < 1
    assert scores["class_weight"] == "balanced" and scores["seed"] == 42


def test_training_is_repeatable(tmp_path):  # TC-TR-07
    csv_path = write_csv(tmp_path / "reports.csv")
    first = run_training(csv_path, str(tmp_path / "m1"))
    second = run_training(csv_path, str(tmp_path / "m2"))
    assert first["best_model"] == second["best_model"]
    assert first["test_scores"]["f1_macro"] == second["test_scores"]["f1_macro"]


def test_sample_size_limits_rows(tmp_path):  # TC-TR-08
    csv_path = write_csv(tmp_path / "reports.csv")
    report = run_training(csv_path, str(tmp_path / "models"), sample_size=100)
    assert report["data_info"]["rows_used"] <= 100


def test_no_class_weight_still_works(tmp_path):  # TC-TR-09
    csv_path = write_csv(tmp_path / "reports.csv")
    report = run_training(csv_path, str(tmp_path / "models"), class_weight=None)
    assert report["test_scores"]["accuracy"] > 0.5


def test_with_confidence_saves_a_model_that_gives_confidence(tmp_path):  # TC-TR-10
    csv_path = write_csv(tmp_path / "reports.csv")
    report = run_training(csv_path, str(tmp_path / "models"), with_confidence=True)
    assert report["best_model"] in ["logistic_regression", "naive_bayes", "linear_svm_calibrated"]
    result = predict(load_model(report["model_path"]), ["crash exception broken"])
    assert 0 <= result.loc[0, "confidence"] <= 1
    with open(os.path.join(str(tmp_path / "models"), "test_scores.json"), encoding="utf-8") as f:
        assert json.load(f)["with_confidence"] is True
