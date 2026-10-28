"""classifier.py - Train, save, load and use a bug report classifier.

A "pipeline" joins two steps: TF-IDF (turns text into numbers) and a model
(learns which numbers mean which component). Saving the whole pipeline keeps
the preprocessing and the model together.

This file does NOT measure quality. That is the job of evaluation.py (next module).
"""

import os

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

RANDOM_SEED = 42  # fixed so results are the same every run
MODEL_NAMES = ["logistic_regression", "naive_bayes", "linear_svm"]
# Linear SVM cannot give a confidence score by itself. "linear_svm_calibrated" wraps it
# so it can. CONFIDENCE_MODEL_NAMES are the models that can show a confidence.
CONFIDENCE_MODEL_NAMES = ["logistic_regression", "naive_bayes", "linear_svm_calibrated"]
ALL_MODEL_NAMES = MODEL_NAMES + ["linear_svm_calibrated"]


class ClassifierError(Exception):
    """Raised for invalid training data, bad model names, or missing model files."""


def build_pipeline(model_name="logistic_regression", class_weight=None):
    """Create an untrained pipeline: TF-IDF + the chosen model.

    class_weight="balanced" makes rare classes count more during training.
    Naive Bayes does not support it and ignores it.
    """
    if model_name == "logistic_regression":
        model = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED, class_weight=class_weight)
    elif model_name == "naive_bayes":
        model = MultinomialNB()
    elif model_name == "linear_svm":
        model = LinearSVC(random_state=RANDOM_SEED, class_weight=class_weight)
    elif model_name == "linear_svm_calibrated":
        # cv=3: the calibration is learned on held-out parts of the training data.
        model = CalibratedClassifierCV(LinearSVC(random_state=RANDOM_SEED, class_weight=class_weight), cv=3)
    else:
        raise ClassifierError(f"Unknown model '{model_name}'. Choose one of: {ALL_MODEL_NAMES}")
    return Pipeline([("tfidf", TfidfVectorizer()), ("model", model)])


def train_classifier(texts, labels, model_name="logistic_regression", class_weight=None):
    """Train a pipeline on cleaned texts and their component labels."""
    texts, labels = list(texts), list(labels)
    if len(texts) == 0:
        raise ClassifierError("No training data was given.")
    if len(texts) != len(labels):
        raise ClassifierError("texts and labels must have the same length.")
    if any((not isinstance(label, str)) or label.strip() == "" for label in labels):
        raise ClassifierError("All labels must be non-empty text.")
    if len(set(labels)) < 2:
        raise ClassifierError("Training needs at least 2 different classes.")
    if all((not isinstance(t, str)) or t.strip() == "" for t in texts):
        raise ClassifierError("All training texts are empty.")

    pipeline = build_pipeline(model_name, class_weight)
    pipeline.fit([t if isinstance(t, str) else "" for t in texts], labels)
    return pipeline


def predict(pipeline, texts):
    """Predict a component for each text.

    Returns a DataFrame with columns predicted_component and confidence.
    confidence is a number from 0 to 1, or None when the model cannot give one
    (Linear SVM).
    """
    texts = [t if isinstance(t, str) else "" for t in texts]
    if len(texts) == 0:
        return pd.DataFrame({"predicted_component": [], "confidence": []})

    predicted = pipeline.predict(texts)
    model = pipeline.named_steps["model"]
    if hasattr(model, "predict_proba"):
        confidence = pipeline.predict_proba(texts).max(axis=1)
    else:
        confidence = [None] * len(texts)
    return pd.DataFrame({"predicted_component": predicted, "confidence": confidence})


def save_model(pipeline, path):
    """Save the trained pipeline to a file (creates the folder if needed)."""
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    joblib.dump(pipeline, path)


def load_model(path):
    """Load a pipeline saved with save_model."""
    if not os.path.exists(path):
        raise ClassifierError(f"Model file not found: {path}")
    return joblib.load(path)
