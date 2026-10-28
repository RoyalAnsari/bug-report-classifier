"""train_model.py - Training script. Run it on your computer; the app only LOADS the result.

Steps: load CSV -> clean text -> remove repeated texts -> split train/val/test
       -> compare 3 models on validation -> score the best one ONCE on test -> save.

Run:
    python -m src.train_model                 (full dataset)
    python -m src.train_model --sample 20000  (quick trial run)
"""

import argparse
import json
import os

from src.classifier import CONFIDENCE_MODEL_NAMES, RANDOM_SEED, save_model
from src.data_loader import load_bug_reports
from src.evaluation import (
    compare_models,
    evaluate_model,
    majority_baseline_accuracy,
    select_best,
    split_train_val_test,
)
from src.preprocessing import preprocess_reports


def prepare_training_data(df):
    """Take the preprocessed table and return (texts, labels, info).

    - removes rows without a label
    - removes rows whose cleaned text is a repeat (the same title in train and
      test would let the model "cheat"); the first row is kept
    """
    if "component" not in df.columns or "clean_text" not in df.columns:
        raise ValueError("Table needs the columns 'component' and 'clean_text'.")

    data = df.copy()
    data["component"] = data["component"].fillna("").astype(str).str.strip()

    no_label = data["component"] == ""
    data = data[~no_label]
    repeated = data.duplicated(subset=["clean_text"])
    data = data[~repeated]

    info = {
        "rows_without_label_removed": int(no_label.sum()),
        "repeated_texts_removed": int(repeated.sum()),
        "rows_used": len(data),
    }
    return list(data["clean_text"]), list(data["component"]), info


def run_training(csv_path="data/bug_reports.csv", model_dir="models", class_weight="balanced",
                 stem=False, sample_size=None, seed=RANDOM_SEED, with_confidence=False):
    """Run the whole training workflow. Returns a report (dict) and saves files in model_dir.

    with_confidence=True only tries models that can show a confidence score.
    """
    df, load_warnings = load_bug_reports(csv_path)
    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=seed)
    clean, prep_warnings = preprocess_reports(df, stem=stem)
    texts, labels, info = prepare_training_data(clean)

    parts = split_train_val_test(texts, labels, val_size=0.15, test_size=0.15, seed=seed)
    model_names = CONFIDENCE_MODEL_NAMES if with_confidence else None
    results, models = compare_models(parts["train"], parts["val"], model_names=model_names, class_weight=class_weight)
    best_name = select_best(results)

    # The test set is used only here, once, for the chosen model.
    test_scores = evaluate_model(models[best_name], *parts["test"])
    baseline = majority_baseline_accuracy(parts["train"][1], parts["test"][1])

    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "classifier.joblib")
    save_model(models[best_name], model_path)
    results.to_csv(os.path.join(model_dir, "model_comparison.csv"), index=False)
    test_scores["confusion_matrix"].to_csv(os.path.join(model_dir, "test_confusion_matrix.csv"))
    summary = {k: float(v) for k, v in test_scores.items() if k != "confusion_matrix"}
    summary.update({"best_model": best_name, "majority_baseline_accuracy": baseline,
                    "class_weight": class_weight, "stemming": stem, "seed": seed,
                    "with_confidence": with_confidence})
    with open(os.path.join(model_dir, "test_scores.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return {
        "warnings": load_warnings + prep_warnings,
        "data_info": info,
        "split_sizes": {name: len(parts[name][0]) for name in parts},
        "comparison": results,
        "best_model": best_name,
        "test_scores": test_scores,
        "majority_baseline_accuracy": baseline,
        "model_path": model_path,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=None, help="use only this many rows (quick trial)")
    parser.add_argument("--stem", action="store_true", help="apply stemming (needs NLTK)")
    parser.add_argument("--with-confidence", action="store_true", help="only use models that give a confidence score")
    parser.add_argument("--no-balance", action="store_true", help="do not use balanced class weights")
    args = parser.parse_args()

    print("Training... (the full dataset can take a few minutes)")
    report = run_training(sample_size=args.sample, stem=args.stem,
                          class_weight=None if args.no_balance else "balanced",
                          with_confidence=args.with_confidence)
    print("\nWarnings:", report["warnings"])
    print("Data:", report["data_info"])
    print("Split sizes:", report["split_sizes"])
    print("\nModel comparison on VALIDATION data:")
    print(report["comparison"].round(4).to_string(index=False))
    scores = report["test_scores"]
    print(f"\nBest model: {report['best_model']}")
    print("FINAL scores on TEST data:")
    for name in ["accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted"]:
        print(f"  {name}: {scores[name]:.4f}")
    print(f"  (always predicting the most common class gets accuracy {report['majority_baseline_accuracy']:.4f})")
    print("\nConfusion matrix (rows = true, columns = predicted):")
    print(scores["confusion_matrix"].to_string())
    print("\nSaved model to", report["model_path"])


if __name__ == "__main__":
    main()
