"""evaluation.py - Split data, measure model quality, and compare models.

Three data sets are used (this prevents "data leakage"):
    train      - the model learns from this
    validation - we compare models on this and pick the best one
    test       - used ONCE at the end for the final score; never used to choose a model
"""

import math
from collections import Counter

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from src.classifier import MODEL_NAMES, RANDOM_SEED, train_classifier

METRIC_COLUMNS = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted"]


class EvaluationError(Exception):
    """Raised when the data given for splitting or evaluation is not valid."""


def can_stratify(labels, test_size):
    """Stratified splitting keeps class proportions. It needs every class to have
    at least 2 examples, and enough room in both parts for every class."""
    counts = Counter(labels)
    n_test = math.ceil(test_size * len(labels))
    n_train = len(labels) - n_test
    n_classes = len(counts)
    return min(counts.values()) >= 2 and n_test >= n_classes and n_train >= n_classes


def split_data(texts, labels, test_size=0.2, seed=RANDOM_SEED):
    """Split into two parts. Returns (texts_a, texts_b, labels_a, labels_b).

    Part b has about test_size of the data. The split keeps class proportions when
    possible (see can_stratify); otherwise it is a plain random split.
    """
    texts, labels = list(texts), list(labels)
    if len(texts) == 0:
        raise EvaluationError("No data to split.")
    if len(texts) != len(labels):
        raise EvaluationError("texts and labels must have the same length.")
    if not 0 < test_size < 1:
        raise EvaluationError("test_size must be between 0 and 1.")
    if math.ceil(test_size * len(texts)) < 1 or len(texts) < 2:
        raise EvaluationError("Not enough data to make a split.")

    stratify = labels if can_stratify(labels, test_size) else None
    return train_test_split(
        texts, labels, test_size=test_size, random_state=seed, stratify=stratify
    )


def split_train_val_test(texts, labels, val_size=0.15, test_size=0.15, seed=RANDOM_SEED):
    """Split into train, validation and test sets.

    Returns a dict: {"train": (texts, labels), "val": (...), "test": (...)}.
    """
    if val_size <= 0 or test_size <= 0 or val_size + test_size >= 1:
        raise EvaluationError("val_size and test_size must be above 0 and add up to less than 1.")
    rest_x, test_x, rest_y, test_y = split_data(texts, labels, test_size, seed)
    # val_size is a share of the WHOLE data, so rescale it for the remaining part.
    val_share = val_size / (1 - test_size)
    train_x, val_x, train_y, val_y = split_data(rest_x, rest_y, val_share, seed)
    return {"train": (train_x, train_y), "val": (val_x, val_y), "test": (test_x, test_y)}


def evaluate_model(pipeline, texts, labels):
    """Measure a trained pipeline on texts with known labels.

    Returns a dict with: accuracy, precision_macro, recall_macro, f1_macro,
    f1_weighted, and confusion_matrix (a DataFrame: rows = true class,
    columns = predicted class).

    "macro" averages every class equally (rare classes count as much as common ones).
    "weighted" averages by class size.
    """
    texts, labels = list(texts), list(labels)
    if len(texts) == 0:
        raise EvaluationError("No data to evaluate.")
    if len(texts) != len(labels):
        raise EvaluationError("texts and labels must have the same length.")

    class_names = list(pipeline.classes_)
    predicted = pipeline.predict([t if isinstance(t, str) else "" for t in texts])

    # zero_division=0: a class that is never predicted gets score 0 instead of a warning.
    scores = {
        "accuracy": accuracy_score(labels, predicted),
        "precision_macro": precision_score(labels, predicted, labels=class_names, average="macro", zero_division=0),
        "recall_macro": recall_score(labels, predicted, labels=class_names, average="macro", zero_division=0),
        "f1_macro": f1_score(labels, predicted, labels=class_names, average="macro", zero_division=0),
        "f1_weighted": f1_score(labels, predicted, labels=class_names, average="weighted", zero_division=0),
    }
    matrix = confusion_matrix(labels, predicted, labels=class_names)
    scores["confusion_matrix"] = pd.DataFrame(matrix, index=class_names, columns=class_names)
    return scores


def compare_models(train_data, val_data, model_names=None, class_weight=None):
    """Train each model on train_data and score it on val_data.

    train_data and val_data are (texts, labels) pairs.
    class_weight="balanced" helps when some classes are rare.
    Returns (results, models):
        results - DataFrame, one row per model, best f1_macro first
        models  - dict {model_name: trained pipeline}
    """
    model_names = list(model_names) if model_names else list(MODEL_NAMES)
    models, rows = {}, []
    for name in model_names:
        models[name] = train_classifier(train_data[0], train_data[1], name, class_weight)
        scores = evaluate_model(models[name], val_data[0], val_data[1])
        rows.append({"model": name, **{col: scores[col] for col in METRIC_COLUMNS}})
    results = pd.DataFrame(rows).sort_values("f1_macro", ascending=False).reset_index(drop=True)
    return results, models


def select_best(results, metric="f1_macro"):
    """Return the name of the model with the highest value of metric."""
    if results is None or len(results) == 0:
        raise EvaluationError("No results to choose from.")
    if metric not in results.columns:
        raise EvaluationError(f"Unknown metric '{metric}'.")
    return results.sort_values(metric, ascending=False).iloc[0]["model"]


def majority_baseline_accuracy(train_labels, test_labels):
    """Accuracy of a "model" that always predicts the most common training class.

    A real model must beat this number to be useful.
    """
    train_labels, test_labels = list(train_labels), list(test_labels)
    if len(train_labels) == 0 or len(test_labels) == 0:
        raise EvaluationError("Labels must not be empty.")
    most_common = Counter(train_labels).most_common(1)[0][0]
    return sum(1 for label in test_labels if label == most_common) / len(test_labels)
