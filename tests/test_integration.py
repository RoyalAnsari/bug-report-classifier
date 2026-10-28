"""Integration tests: do the modules work together? (Grows each week.)"""

from src.data_loader import load_bug_reports
from src.preprocessing import preprocess_reports


def test_loader_to_preprocessing_sample_file():  # TC-INT-01
    df, load_warnings = load_bug_reports("data/sample_bug_reports.csv")
    result, prep_warnings = preprocess_reports(df)
    assert len(result) == len(df) == 8
    assert "clean_text" in result.columns
    assert (result["clean_text"] != "").all()
    assert load_warnings == [] and prep_warnings == []


def test_loader_to_preprocessing_messy_file(tmp_path):  # TC-INT-02
    path = tmp_path / "messy.csv"
    path.write_text(
        "bug_id,title,description\n1,  APP Crash!! ,See https://x.com\n2,???,\n1,Dup id,x\n",
        encoding="utf-8",
    )
    df, _ = load_bug_reports(str(path))
    result, _ = preprocess_reports(df)
    assert list(result["bug_id"]) == ["1"]
    assert result.loc[0, "clean_text"] == "app crash see"


def test_preprocessing_to_classifier_and_saved_model(tmp_path):  # TC-INT-03
    from src.classifier import load_model, predict, save_model, train_classifier

    df, _ = load_bug_reports("data/sample_bug_reports.csv")
    clean, _ = preprocess_reports(df)
    model = train_classifier(clean["clean_text"], clean["component"])
    path = str(tmp_path / "model.joblib")
    save_model(model, path)
    result = predict(load_model(path), clean["clean_text"])
    assert len(result) == len(clean)
    assert set(result["predicted_component"]) <= set(clean["component"])


def test_dataset_prep_to_loader_to_preprocessing(tmp_path):  # TC-INT-04
    import pandas as pd

    from src.dataset_prep import prepare_bug_reports

    raw = pd.DataFrame(
        {
            "Project": ["Bugzilla", "Eclipse", "Eclipse"],
            "Bug ID": [1, 1, 2],
            "Severity Label": ["normal", "major", "minor"],
            "Resolution Status": ["FIXED"] * 3,
            "Short Description": ["Crash on login!", "NPE in editor", "???"],
        }
    )
    path = tmp_path / "bug_reports.csv"
    prepare_bug_reports(raw).to_csv(path, index=False)

    df, load_warnings = load_bug_reports(str(path))
    assert len(df) == 3  # ids stay unique because Project is part of bug_id
    assert any("empty description" in w for w in load_warnings)

    clean, _ = preprocess_reports(df)
    assert list(clean["bug_id"]) == ["Bugzilla-1", "Eclipse-1"]  # "???" removed
    assert list(clean["component"]) == ["normal", "major"]


def test_full_training_workflow(tmp_path):  # TC-INT-05
    """CSV -> loader -> preprocessing -> split -> compare -> best model -> test score."""
    from src.classifier import predict
    from src.evaluation import compare_models, evaluate_model, select_best, split_train_val_test

    words = {"minor": "typo spelling label text", "major": "crash exception fails broken", "normal": "update docs page link"}
    lines = ["bug_id,title,description,component"]
    n = 0
    for label, w in words.items():
        for i in range(20):
            n += 1
            lines.append(f"{n},{w} item{i}!,,{label}")
    path = tmp_path / "reports.csv"
    path.write_text("\n".join(lines), encoding="utf-8")

    df, _ = load_bug_reports(str(path))
    clean, _ = preprocess_reports(df)
    parts = split_train_val_test(clean["clean_text"], clean["component"], 0.2, 0.2)
    results, models = compare_models(parts["train"], parts["val"])
    best = select_best(results)
    final = evaluate_model(models[best], *parts["test"])
    assert 0 <= final["f1_macro"] <= 1
    assert len(predict(models[best], parts["test"][0])) == len(parts["test"][0])


def test_preprocessing_to_duplicate_detection(tmp_path):  # TC-INT-06
    from src.duplicate_detector import find_duplicate_pairs, find_similar_reports

    path = tmp_path / "reports.csv"
    path.write_text(
        "bug_id,title,description\n"
        "Eclipse-1,NPE in Java editor on save!,\n"
        "Eclipse-2,Java editor NPE when saving,\n"
        "Eclipse-3,Typo in preferences dialog,\n"
        "Eclipse-4,Preferences dialog has a typo,\n"
        "Eclipse-5,Build is very slow,\n",
        encoding="utf-8",
    )
    df, _ = load_bug_reports(str(path))
    clean, _ = preprocess_reports(df)

    pairs = find_duplicate_pairs(clean["clean_text"], clean["bug_id"], threshold=0.3)
    found = {frozenset((r.query_id, r.match_id)) for r in pairs.itertuples()}
    assert frozenset(("Eclipse-1", "Eclipse-2")) in found
    assert frozenset(("Eclipse-3", "Eclipse-4")) in found
    assert not any("Eclipse-5" in pair for pair in found)

    # A new report is compared with the existing ones.
    new = find_similar_reports(["java editor npe on save"], ["New-1"], clean["clean_text"], clean["bug_id"], threshold=0.3)
    assert new.loc[0, "match_id"] in ("Eclipse-1", "Eclipse-2")


def test_duplicates_to_human_review_and_export(tmp_path):  # TC-INT-07
    """Duplicate suggestions -> human decisions -> save -> load -> new threshold keeps decisions."""
    from src.duplicate_detector import find_duplicate_pairs
    from src.review import create_review_table, load_review, merge_previous_decisions, review_summary, save_review, set_decision

    path = tmp_path / "reports.csv"
    path.write_text(
        "bug_id,title,description\n"
        "1,NPE in Java editor on save,\n2,Java editor NPE when saving on save,\n"
        "3,Typo in preferences dialog,\n4,Preferences dialog typo,\n5,Build is very slow,\n",
        encoding="utf-8",
    )
    df, _ = load_bug_reports(str(path))
    clean, _ = preprocess_reports(df)

    pairs = find_duplicate_pairs(clean["clean_text"], clean["bug_id"], threshold=0.3)
    table = create_review_table(pairs)
    assert review_summary(table)["pending"] == len(table) >= 2

    table = set_decision(table, "1", "2", "confirmed")
    table = set_decision(table, "3", "4", "rejected")
    assert (table["status"] == "potential duplicate").all()  # machine label untouched

    out = str(tmp_path / "decisions.csv")
    save_review(table, out)
    loaded = load_review(out)
    assert review_summary(loaded)["confirmed"] == 1 and review_summary(loaded)["rejected"] == 1

    # The user moves the threshold: new suggestions, but old decisions are kept.
    new_pairs = find_duplicate_pairs(clean["clean_text"], clean["bug_id"], threshold=0.1)
    merged = merge_previous_decisions(create_review_table(new_pairs), loaded)
    assert review_summary(merged)["confirmed"] == 1 and review_summary(merged)["rejected"] == 1


def test_train_then_pipeline_then_review(tmp_path):  # TC-INT-08
    """Training script output -> pipeline -> human review: the full backend workflow."""
    from src.pipeline import add_titles, run_pipeline
    from src.review import review_summary, set_decision
    from src.train_model import run_training

    words = {"minor": "typo spelling label text", "major": "crash exception fails broken", "normal": "update docs page link"}
    lines = ["bug_id,title,description,component"]
    n = 0
    for label, w in words.items():
        for i in range(25):
            n += 1
            lines.append(f"{n},{w} item{i},,{label}")
    train_csv = tmp_path / "train.csv"
    train_csv.write_text("\n".join(lines), encoding="utf-8")
    model_dir = str(tmp_path / "models")
    run_training(str(train_csv), model_dir)

    upload = tmp_path / "upload.csv"
    upload.write_text(
        "bug_id,title,description\nU1,App crash exception broken,\nU2,Crash exception broken in app,\nU3,Typo in label text,\n",
        encoding="utf-8",
    )
    result = run_pipeline(str(upload), model_path=model_dir + "/classifier.joblib", threshold=0.3)
    assert len(result["reports"]) == 3
    assert len(result["duplicates"]) >= 1

    shown = add_titles(result["duplicates"], result["reports"])
    first = shown.iloc[0]
    reviewed = set_decision(result["duplicates"], first["query_id"], first["match_id"], "confirmed")
    assert review_summary(reviewed)["confirmed"] == 1


def test_pipeline_to_reporting_and_export(tmp_path):  # TC-INT-09
    """Pipeline result -> search/filter/sort -> review -> charts data -> CSV export."""
    import io

    import pandas as pd

    from src.classifier import save_model, train_classifier
    from src.pipeline import add_titles, run_pipeline
    from src.reporting import (
        class_distribution, decision_distribution, duplicates_to_csv, filter_pairs,
        filter_reports, overview_metrics, reports_to_csv, search_reports, sort_reports,
    )
    from src.review import set_decision

    texts = ["crash exception broken"] * 6 + ["typo label text"] * 6
    labels = ["major"] * 6 + ["trivial"] * 6
    model_path = str(tmp_path / "classifier.joblib")
    save_model(train_classifier(texts, labels), model_path)

    csv = (
        "bug_id,title,description\n"
        "1,App crash exception broken,\n2,Crash exception broken in app,\n3,Typo in label text,\n"
    )
    result = run_pipeline(io.StringIO(csv), model_path, threshold=0.3)
    reports = result["reports"]

    assert len(search_reports(reports, "crash")) == 2
    assert set(filter_reports(reports, predicted_classes=["trivial"])["bug_id"]) == {"3"}
    assert len(sort_reports(reports, "bug_id")) == 3

    pairs = add_titles(result["duplicates"], reports)
    first = pairs.iloc[0]
    pairs = set_decision(pairs, first["query_id"], first["match_id"], "confirmed")
    assert len(filter_pairs(pairs, decisions=["confirmed"])) == 1
    assert overview_metrics(reports, pairs)["confirmed"] == 1
    assert class_distribution(reports)["count"].sum() == 3
    assert decision_distribution(pairs)["count"].sum() == len(pairs)

    exported = pd.read_csv(io.StringIO(duplicates_to_csv(pairs)), dtype=str)
    assert "confirmed" in set(exported["human_decision"])
    assert len(pd.read_csv(io.StringIO(reports_to_csv(reports)))) == 3
