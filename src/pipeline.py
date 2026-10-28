"""pipeline.py - Connects the tested modules into ONE workflow.

    CSV file -> load/validate -> clean text -> classify -> find duplicates -> review table

The Streamlit app (Week 4) will call run_pipeline() and nothing else from the backend.
No new logic lives here: every step is done by a module that has its own tests.
"""

import json
import os

import pandas as pd

from src.classifier import ClassifierError, load_model, predict
from src.data_loader import DataValidationError, load_bug_reports
from src.duplicate_detector import (
    DEFAULT_THRESHOLD,
    DEFAULT_TOP_K,
    DuplicateDetectorError,
    find_duplicate_pairs,
)
from src.preprocessing import preprocess_reports
from src.review import create_review_table

DEFAULT_MODEL_PATH = "models/classifier.joblib"
DEFAULT_MAX_REPORTS = 20000  # protects the app from very slow runs


class PipelineError(Exception):
    """One error type for the app to catch. The message is safe to show to the user."""


def _model_uses_stemming(model_path):
    """The training script saves test_scores.json next to the model.
    It says whether training used stemming; new reports must be cleaned the same way."""
    scores_path = os.path.join(os.path.dirname(model_path), "test_scores.json")
    try:
        with open(scores_path, encoding="utf-8") as f:
            return bool(json.load(f).get("stemming", False))
    except (FileNotFoundError, ValueError):
        return False


def run_pipeline(source, model_path=DEFAULT_MODEL_PATH, threshold=DEFAULT_THRESHOLD,
                 top_k=DEFAULT_TOP_K, stem=None, max_reports=DEFAULT_MAX_REPORTS):
    """Run the whole backend workflow on an uploaded CSV (path or file object).

    stem=None means "do what the saved model was trained with".

    Returns a dict:
        reports    - DataFrame: one row per report, with clean_text,
                     predicted_component and confidence
        duplicates - review table of possible duplicate pairs (all 'pending')
        warnings   - list of messages for the user
        summary    - counts (reports, predicted classes, duplicate pairs)
    Raises PipelineError with a clear message for any known problem.
    """
    try:
        df, warnings = load_bug_reports(source)
        if max_reports and len(df) > max_reports:
            raise PipelineError(
                f"The file has {len(df)} reports; the limit is {max_reports}. Please upload a smaller file."
            )

        model = load_model(model_path) if os.path.exists(model_path) else None
        if model is None:
            raise PipelineError(
                f"Model file not found: {model_path}. Train it first with: python -m src.train_model"
            )

        if stem is None:
            stem = _model_uses_stemming(model_path)
        clean, prep_warnings = preprocess_reports(df, stem=stem)
        if len(clean) == 0:
            raise PipelineError("No usable reports remain after cleaning.")

        predictions = predict(model, clean["clean_text"])
        reports = pd.concat([clean.reset_index(drop=True), predictions], axis=1)

        pairs = find_duplicate_pairs(reports["clean_text"], reports["bug_id"], threshold, top_k)
        duplicates = create_review_table(pairs)
    except (DataValidationError, ClassifierError, DuplicateDetectorError) as error:
        raise PipelineError(str(error)) from error

    summary = {
        "reports": len(reports),
        "duplicate_pairs": len(duplicates),
        "predicted_classes": reports["predicted_component"].value_counts().to_dict(),
    }
    return {"reports": reports, "duplicates": duplicates,
            "warnings": warnings + prep_warnings, "summary": summary}


def add_titles(duplicates, reports):
    """Add query_title and match_title to a review table, so people can read the pairs."""
    titles = dict(zip(reports["bug_id"], reports["title"]))
    result = duplicates.copy()
    result.insert(2, "query_title", result["query_id"].map(titles))
    result.insert(4, "match_title", result["match_id"].map(titles))
    return result
