# Bug Report Classifier and Duplicate Detector

A small web application for QA teams. You upload a CSV file of bug reports. The app:

1. **Classifies** each report by severity (blocker, critical, major, normal, minor, trivial).
2. **Suggests possible duplicates** (reports that describe the same problem in similar words).
3. Lets a **person confirm or reject** every suggested duplicate.
4. Shows charts, search and filters, and **exports the results to CSV**.

The machine only suggests. A person always decides. Machine output and human decisions are stored in separate columns.

Built with Python, scikit-learn, Streamlit and Plotly. It uses a public dataset, not private company tickets.

## Screenshots
Add your screenshots to `docs/screenshots/` and list them here after you run the app:
- Welcome page
- Overview tab (summary cards and charts)
- Duplicate review tab
- Export tab

## Features
| Feature | Where |
|---|---|
| CSV upload and validation with clear error messages | `src/data_loader.py` |
| Text cleaning (lowercase, remove links and symbols, optional stemming) | `src/preprocessing.py` |
| Severity classification: TF-IDF with Logistic Regression, Naive Bayes, Linear SVM | `src/classifier.py`, `src/evaluation.py`, `src/train_model.py` |
| Confidence score (only for models that support it) | `src/classifier.py` |
| Duplicate suggestions: TF-IDF with cosine similarity and an adjustable threshold | `src/duplicate_detector.py` |
| Human review: confirm, reject, undo; decisions are saved | `src/review.py` |
| Search, filter, sort, chart data, CSV export | `src/reporting.py` |
| One function joining all steps | `src/pipeline.py` |
| Streamlit interface | `app.py` |

## Dataset
The project uses the Hugging Face dataset **`blak002/Bugzilla_Eclipse_Bug_Reports_Dataset`**, a copy of `AliArshad/Bugzilla_Eclipse_Bug_Reports_Dataset`. It is a processed part of the Eclipse and Mozilla defect tracking dataset. Please cite the original paper:

> Lamkanfi, A., Perez, J., Demeyer, S. (2013). *The Eclipse and Mozilla defect tracking dataset: a genuine dataset for mining bug information.* 10th Working Conference on Mining Software Repositories (MSR). doi:10.1109/MSR.2013.6624028

What the dataset contains (columns): `Project`, `Bug ID`, `Severity Label`, `Resolution Status`, `Short Description`. It has about 88.7 thousand rows.

Important facts:
- There is **only a short title**, no long description.
- There is **no component column**. This project predicts the **severity**.
- There are **no "duplicate of" links**, so duplicate quality cannot be scored automatically.
- The dataset card does not state a licence. The data is **not included** in this repository. Check the licence before sharing it.

`src/dataset_prep.py` converts the dataset to this project's format (`bug_id`, `title`, `description`, `component`). The severity is stored in the `component` column, and the project name is added to the bug id because different projects reuse the same numbers.

## Setup (Windows, VS Code terminal, inside this folder)
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Get the data and train the model (once)
```
python -m src.dataset_prep
python -m src.train_model
```
Useful options for training:
| Option | Meaning |
|---|---|
| `--sample 20000` | Use only 20,000 rows (quick trial) |
| `--stem` | Apply stemming (needs NLTK) |
| `--no-balance` | Do not use balanced class weights |
| `--with-confidence` | Only try models that give a confidence score |

Training writes these files to `models/`: `classifier.joblib` (the model), `model_comparison.csv`, `test_confusion_matrix.csv` and `test_scores.json`. Every run overwrites them, so copy the folder first if you want to keep an old result.

## Run the app
```
python -m streamlit run app.py
```
Press **Try the demo file** on the first page, or upload your own CSV in the sidebar.

A manual check of duplicate suggestions on the real dataset (prints pairs of titles):
```
python try_duplicates.py
```

## Input file format
A UTF-8 CSV file with a header row.

| Column | Required | Notes |
|---|---|---|
| `bug_id` | Yes | Text, unique |
| `title` | Yes | Short summary |
| `description` | Yes | May be empty if the title exists |
| `component` | No | The true label from your file, used for filtering |

Rows with no id or no text, and repeated ids, are removed with a warning. In Excel, use Save As and choose **CSV UTF-8**.

## How it works
```
CSV -> validate -> clean text -> classify -> find duplicates -> human review -> charts / export
```
- **Training data split:** 70% train, 15% validation, 15% test. Models are compared on validation. The test set is used once, for the chosen model.
- **Imbalanced classes:** about 81% of the reports are "normal", so the models use balanced class weights and are compared by **macro F1** (every class counts equally), not by accuracy.
- **Repeated titles** are removed before splitting, so the same title cannot appear in both train and test.
- **Random seed** is fixed (42), so results repeat.

## Results (recorded from a training run on the full dataset, no stemming)
These numbers come from the author's own run and are **not** guaranteed for other data or settings. If you retrain, your numbers may differ.

| Measure (test set, 13,195 reports) | Value |
|---|---|
| Best model (chosen on validation) | Linear SVM |
| Macro F1 | 0.352 |
| Accuracy | 0.737 |
| Accuracy of always predicting "normal" | 0.814 |

The model finds **critical** bugs reasonably well (about 68% of them) and is weak on major, minor and trivial. Linear SVM does not give a confidence score; train with `--with-confidence` to get one. A run with stemming gave almost the same macro F1 (0.360 on a slightly different test set), so stemming is not used.

There are no accuracy figures for duplicate detection, because the dataset has no duplicate labels. Check the suggestions by eye.

## Tests
```
python -m pytest -v
```
Build the test documents from real results:
```
python -m pytest -v --junitxml=docs/test_results.xml
python -m tools.doc_builder cases
python -m tools.doc_builder report
```
This writes `docs/test_cases.md` and `docs/test_report.md`. Test levels: unit, integration, end-to-end, and regression (frozen function interfaces). The Streamlit screen itself is checked with the manual tests in `docs/test_plan.md`.

## Project structure
```
bug-report-classifier/
    app.py                     Streamlit interface (shows things only)
    requirements.txt
    README.md
    pytest.ini
    try_duplicates.py          manual check of duplicate suggestions
    .streamlit/config.toml     theme and settings
    data/
        sample_bug_reports.csv    tiny test file
        demo_bug_reports.csv      20 made-up reports with 4 duplicate pairs
        bug_reports.csv           created by dataset_prep (not in Git)
    models/                    created by training (not in Git)
    src/
        data_loader.py  dataset_prep.py  preprocessing.py  classifier.py
        evaluation.py   train_model.py   duplicate_detector.py
        review.py       pipeline.py      reporting.py
    tests/                     unit, integration, end-to-end, interface tests
    tools/doc_builder.py       builds test documents from real test results
    docs/
        FRD.md  test_plan.md  test_cases.md  bug_log.md
```
Planned for the final documents: `test_report.md` (generated), SRS and `final_report.md`.

## Known limitations
- Only a **short title** is available, so severity is hard to predict. Expect modest results.
- Duplicate detection uses **shared words** (TF-IDF). It misses reports with the same meaning but different words, and it can score two different reports highly when only a small detail differs (for example "X64" and "ARM", or two version numbers). A person must check each suggestion.
- Only **English letters and digits** are used. Text in other languages, such as Urdu, is removed during cleaning, and a report with no English text is dropped with a warning.
- Files must be **UTF-8** and are limited to 20,000 reports per analysis.
- Review decisions are saved in one local file (`data/review_decisions.csv`), shared by everyone who uses that copy of the app. There are no user accounts.
- The model is trained on Eclipse and Mozilla titles. It may not suit other projects.

## Future improvements
Sentence embeddings (Sentence Transformers) for duplicates with different wording, a second dataset with duplicate links to measure duplicate quality, user accounts, and a long description field.

## Deployment (optional)
The app can run on Azure App Service (Linux). It needs a paid plan (B1 or higher) because Streamlit needs websockets, a `requirements.txt` with the same package versions that trained the model, and the startup command:
```
python -m streamlit run app.py --server.port 8000 --server.address 0.0.0.0 --server.headless true
```
Live address: add it here after you deploy.
