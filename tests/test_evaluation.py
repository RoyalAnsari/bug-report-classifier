"""Unit tests for src/evaluation.py (requirements FR-10, FR-12)."""

import pytest

from src.evaluation import (
    METRIC_COLUMNS,
    EvaluationError,
    can_stratify,
    compare_models,
    evaluate_model,
    select_best,
    split_data,
    split_train_val_test,
)
from src.classifier import train_classifier

WORDS = {"UI": "screen button layout window", "Auth": "login password signin token", "DB": "query table index timeout"}


def make_data(per_class=20):
    """Easy, separable data: every text has words from its own class."""
    texts, labels = [], []
    for label, words in WORDS.items():
        for i in range(per_class):
            texts.append(f"{words} case{i}")
            labels.append(label)
    return texts, labels


# ---------- split_data ----------
def test_split_sizes():  # TC-EV-01
    texts, labels = make_data()
    a, b, la, lb = split_data(texts, labels, test_size=0.2)
    assert len(a) + len(b) == 60 and len(b) == 12
    assert len(la) == len(a) and len(lb) == len(b)


def test_split_has_no_overlap():  # TC-EV-02 (every text here is unique)
    texts, labels = make_data()
    a, b, _, _ = split_data(texts, labels)
    assert set(a).isdisjoint(set(b))


def test_split_keeps_all_classes():  # TC-EV-03
    texts, labels = make_data()
    _, _, la, lb = split_data(texts, labels)
    assert set(la) == set(lb) == set(WORDS)


def test_split_is_repeatable():  # TC-EV-04
    texts, labels = make_data()
    assert split_data(texts, labels)[1] == split_data(texts, labels)[1]


def test_rare_class_does_not_crash():  # TC-EV-05
    texts, labels = make_data()
    texts.append("rare bug")
    labels.append("Rare")
    assert can_stratify(labels, 0.2) is False
    a, b, _, _ = split_data(texts, labels)
    assert len(a) + len(b) == 61


def test_split_invalid_input():  # TC-EV-06
    with pytest.raises(EvaluationError, match="No data"):
        split_data([], [])
    with pytest.raises(EvaluationError, match="same length"):
        split_data(["a", "b"], ["x"])
    with pytest.raises(EvaluationError, match="between 0 and 1"):
        split_data(["a", "b"], ["x", "y"], test_size=1.5)
    with pytest.raises(EvaluationError, match="Not enough"):
        split_data(["a"], ["x"])


# ---------- split_train_val_test ----------
def test_three_way_split():  # TC-EV-07
    texts, labels = make_data()
    parts = split_train_val_test(texts, labels, val_size=0.2, test_size=0.2)
    sizes = {name: len(parts[name][0]) for name in parts}
    assert sum(sizes.values()) == 60
    assert sizes["test"] == 12 and sizes["val"] == 12 and sizes["train"] == 36
    all_texts = parts["train"][0] + parts["val"][0] + parts["test"][0]
    assert len(set(all_texts)) == 60  # no text appears in two parts


def test_three_way_split_invalid_sizes():  # TC-EV-08
    texts, labels = make_data()
    with pytest.raises(EvaluationError, match="add up"):
        split_train_val_test(texts, labels, val_size=0.6, test_size=0.5)


# ---------- evaluate_model ----------
def test_evaluate_easy_data_scores_high():  # TC-EV-09
    texts, labels = make_data()
    model = train_classifier(texts, labels)
    scores = evaluate_model(model, texts, labels)
    assert scores["accuracy"] > 0.95
    for col in METRIC_COLUMNS:
        assert 0 <= scores[col] <= 1


def test_confusion_matrix_shape_and_total():  # TC-EV-10
    texts, labels = make_data()
    model = train_classifier(texts, labels)
    matrix = evaluate_model(model, texts, labels)["confusion_matrix"]
    assert matrix.shape == (3, 3)
    assert int(matrix.values.sum()) == 60
    assert list(matrix.index) == list(matrix.columns)


def test_evaluate_class_missing_from_test_set():  # TC-EV-11
    texts, labels = make_data()
    model = train_classifier(texts, labels)
    only_ui = [(t, l) for t, l in zip(texts, labels) if l == "UI"]
    scores = evaluate_model(model, [t for t, _ in only_ui], [l for _, l in only_ui])
    assert scores["confusion_matrix"].shape == (3, 3)  # all classes still shown


def test_evaluate_invalid_input():  # TC-EV-12
    texts, labels = make_data()
    model = train_classifier(texts, labels)
    with pytest.raises(EvaluationError, match="No data"):
        evaluate_model(model, [], [])
    with pytest.raises(EvaluationError, match="same length"):
        evaluate_model(model, ["a"], ["UI", "DB"])


# ---------- compare_models / select_best ----------
def test_compare_models_table():  # TC-EV-13
    texts, labels = make_data()
    parts = split_train_val_test(texts, labels, val_size=0.2, test_size=0.2)
    results, models = compare_models(parts["train"], parts["val"])
    assert set(results["model"]) == {"logistic_regression", "naive_bayes", "linear_svm"}
    assert set(models) == set(results["model"])
    assert list(results["f1_macro"]) == sorted(results["f1_macro"], reverse=True)
    assert all(col in results.columns for col in METRIC_COLUMNS)


def test_compare_models_subset():  # TC-EV-14
    texts, labels = make_data()
    parts = split_train_val_test(texts, labels, val_size=0.2, test_size=0.2)
    results, _ = compare_models(parts["train"], parts["val"], ["naive_bayes"])
    assert list(results["model"]) == ["naive_bayes"]


def test_select_best():  # TC-EV-15
    texts, labels = make_data()
    parts = split_train_val_test(texts, labels, val_size=0.2, test_size=0.2)
    results, _ = compare_models(parts["train"], parts["val"])
    assert select_best(results) == results.loc[0, "model"]


def test_select_best_invalid():  # TC-EV-16
    with pytest.raises(EvaluationError, match="No results"):
        select_best(None)
    texts, labels = make_data()
    parts = split_train_val_test(texts, labels, val_size=0.2, test_size=0.2)
    results, _ = compare_models(parts["train"], parts["val"], ["naive_bayes"])
    with pytest.raises(EvaluationError, match="Unknown metric"):
        select_best(results, metric="magic")


# ---------- majority baseline / class_weight ----------
from src.evaluation import majority_baseline_accuracy


def test_majority_baseline():  # TC-EV-17
    assert majority_baseline_accuracy(["a", "a", "b"], ["a", "b", "a", "a"]) == 0.75


def test_majority_baseline_empty_raises():  # TC-EV-18
    with pytest.raises(EvaluationError, match="empty"):
        majority_baseline_accuracy([], ["a"])


def test_compare_models_accepts_class_weight():  # TC-EV-19
    texts, labels = make_data()
    parts = split_train_val_test(texts, labels, val_size=0.2, test_size=0.2)
    results, _ = compare_models(parts["train"], parts["val"], class_weight="balanced")
    assert len(results) == 3
