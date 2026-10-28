"""Unit tests for src/classifier.py (requirements FR-09, FR-10, FR-11, FR-20, FR-21)."""

import pytest

from src.classifier import (
    MODEL_NAMES,
    ClassifierError,
    build_pipeline,
    load_model,
    predict,
    save_model,
    train_classifier,
)

TEXTS = [
    "app crashes on start", "screen freezes and crashes", "button text cut off", "window layout broken",
    "login fails wrong password", "cannot sign in password rejected", "password reset email missing", "sign in error",
]
LABELS = ["UI"] * 4 + ["Auth"] * 4


# ---------- build / train ----------
def test_build_all_models():  # TC-CL-01
    for name in MODEL_NAMES:
        assert build_pipeline(name) is not None


def test_unknown_model_raises():  # TC-CL-02
    with pytest.raises(ClassifierError, match="Unknown model"):
        build_pipeline("magic")


def test_train_and_predict_all_models():  # TC-CL-03
    for name in MODEL_NAMES:
        model = train_classifier(TEXTS, LABELS, name)
        result = predict(model, ["password login error", "app crashes screen"])
        assert list(result["predicted_component"]) == ["Auth", "UI"], name


def test_training_is_repeatable():  # TC-CL-04 (fixed seed)
    first = predict(train_classifier(TEXTS, LABELS), ["login password crash"])
    second = predict(train_classifier(TEXTS, LABELS), ["login password crash"])
    assert first.equals(second)


# ---------- confidence ----------
def test_confidence_between_0_and_1():  # TC-CL-05
    for name in ["logistic_regression", "naive_bayes"]:
        result = predict(train_classifier(TEXTS, LABELS, name), ["login password"])
        assert 0 <= result.loc[0, "confidence"] <= 1


def test_svm_has_no_confidence():  # TC-CL-06
    result = predict(train_classifier(TEXTS, LABELS, "linear_svm"), ["login password"])
    assert result.loc[0, "confidence"] is None


# ---------- invalid training input ----------
def test_empty_training_data_raises():  # TC-CL-07
    with pytest.raises(ClassifierError, match="No training data"):
        train_classifier([], [])


def test_length_mismatch_raises():  # TC-CL-08
    with pytest.raises(ClassifierError, match="same length"):
        train_classifier(["a", "b"], ["UI"])


def test_single_class_raises():  # TC-CL-09
    with pytest.raises(ClassifierError, match="2 different"):
        train_classifier(["a b", "c d"], ["UI", "UI"])


def test_empty_label_raises():  # TC-CL-10
    with pytest.raises(ClassifierError, match="labels"):
        train_classifier(["a b", "c d"], ["UI", ""])


def test_all_empty_texts_raise():  # TC-CL-11
    with pytest.raises(ClassifierError, match="empty"):
        train_classifier(["", " "], ["UI", "Auth"])


# ---------- prediction edge cases ----------
def test_predict_empty_list():  # TC-CL-12
    result = predict(train_classifier(TEXTS, LABELS), [])
    assert len(result) == 0


def test_predict_empty_text_does_not_crash():  # TC-CL-13
    result = predict(train_classifier(TEXTS, LABELS), ["", None])
    assert len(result) == 2


def test_unseen_words_do_not_crash():  # TC-CL-14
    result = predict(train_classifier(TEXTS, LABELS), ["zzzz qqqq"])
    assert result.loc[0, "predicted_component"] in ("UI", "Auth")


# ---------- save / load ----------
def test_save_and_load_give_same_predictions(tmp_path):  # TC-CL-15
    model = train_classifier(TEXTS, LABELS)
    path = str(tmp_path / "models" / "model.joblib")
    save_model(model, path)
    loaded = load_model(path)
    new_texts = ["login password", "screen crash"]
    assert predict(model, new_texts).equals(predict(loaded, new_texts))


def test_load_missing_file_raises(tmp_path):  # TC-CL-16
    with pytest.raises(ClassifierError, match="not found"):
        load_model(str(tmp_path / "missing.joblib"))


# ---------- class_weight (added for imbalanced data) ----------
def test_class_weight_balanced_works_for_all_models():  # TC-CL-17
    for name in MODEL_NAMES:  # naive_bayes ignores class_weight
        model = train_classifier(TEXTS, LABELS, name, class_weight="balanced")
        assert list(predict(model, ["password login"])["predicted_component"]) == ["Auth"], name


def test_class_weight_helps_rare_class():  # TC-CL-18
    texts = ["common bug text"] * 30 + ["rare crash"] * 3
    labels = ["normal"] * 30 + ["critical"] * 3
    balanced = train_classifier(texts, labels, "logistic_regression", class_weight="balanced")
    assert predict(balanced, ["rare crash"]).loc[0, "predicted_component"] == "critical"


# ---------- calibrated SVM (gives a confidence score) ----------
from src.classifier import ALL_MODEL_NAMES, CONFIDENCE_MODEL_NAMES


def test_calibrated_svm_gives_confidence():  # TC-CL-19
    texts = TEXTS * 2  # calibration needs a few examples per class
    labels = LABELS * 2
    model = train_classifier(texts, labels, "linear_svm_calibrated")
    result = predict(model, ["password login error", "app crashes screen"])
    assert list(result["predicted_component"]) == ["Auth", "UI"]
    assert all(0 <= c <= 1 for c in result["confidence"])


def test_confidence_model_list():  # TC-CL-20
    assert "linear_svm" not in CONFIDENCE_MODEL_NAMES
    assert set(CONFIDENCE_MODEL_NAMES) <= set(ALL_MODEL_NAMES)
    assert "linear_svm" in MODEL_NAMES  # the default list is unchanged
