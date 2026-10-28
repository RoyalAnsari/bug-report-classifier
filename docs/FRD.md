# Functional Requirements Document (FRD)
**Project:** Bug Report Classifier and Duplicate Detector
**Version:** 0.1 (Week 1) | **Status:** Draft, updated every week

> Items marked **[TO CONFIRM]** must be checked during development. No dataset statistics or model results are claimed in this document.

## 1. Introduction and Purpose
This document describes what the system must do. It is the reference for development and testing.

## 2. Business Problem and Proposed Solution
**Problem:** QA teams receive many bug reports. Different people describe the same problem in different words. Reading, classifying, and finding duplicates by hand is slow.
**Solution:** A web application where a user uploads a CSV of bug reports. The system classifies each report, suggests possible duplicates, and lets a human confirm or reject each suggestion.

## 3. Objectives
1. Upload and validate bug report CSV files.
2. Classify reports by component using TF-IDF and classic ML models.
3. Suggest possible duplicates using TF-IDF + cosine similarity (baseline).
4. Let a human confirm or reject suggestions.
5. Show results, charts, and export to CSV.
6. Deliver tested code and documentation in 4 weeks.

## 4. Scope and Limitations
**In scope:** CSV upload, preprocessing, classification, duplicate suggestions, human review, search/filter, charts, CSV export, model saving/loading, tests.
**Out of scope:** Live connection to bug trackers, user accounts, private company tickets, deep-learning training.
**Limitations:** Results depend on the public dataset. Duplicates are *potential*, never automatic. Sentence embeddings are optional (time and hardware permitting).

## 5. User Roles and Use Cases
| Role | Use cases |
|---|---|
| QA Engineer | Upload reports, view classes, review duplicate suggestions, export results |
| Project Lead | View class distribution and duplicate summary charts |
| Developer (you) | Retrain and save a model with the training script |

## 6. Functional Requirements
| ID | Requirement | Week |
|---|---|---|
| FR-01 | The system shall accept a CSV file upload. | 1 |
| FR-02 | The system shall require columns `bug_id`, `title`, `description`; `component` is optional. | 1 |
| FR-03 | The system shall show a clear error for empty files, unreadable files, header-only files, and missing columns. | 1 |
| FR-04 | The system shall remove rows with no id or no text, and repeated ids, and report this as warnings. | 1 |
| FR-05 | The system shall display uploaded reports in a table. | 4 |
| FR-06 | The system shall lowercase text, remove extra spaces and noise, and combine title and description. | 1 |
| FR-07 | The system shall handle missing values during preprocessing. | 1 |
| FR-08 | Stemming and lemmatization shall be evaluated and the choice recorded. | 1-2 |
| FR-09 | The system shall classify each report into a component using TF-IDF with a traditional ML model. | 2 |
| FR-10 | Logistic Regression, Naive Bayes, and Linear SVM shall be compared. | 2 |
| FR-11 | The system shall show the predicted class and, where supported, a confidence score. | 2 |
| FR-12 | The system shall report accuracy, precision, recall, F1, and a confusion matrix. | 2 |
| FR-13 | The system shall find similar reports with TF-IDF cosine similarity and show scores. | 3 |
| FR-14 | The user shall be able to set a similarity threshold. | 3 |
| FR-15 | Matches shall be labelled "potential duplicate", never "confirmed". | 3 |
| FR-16 | The user shall confirm or reject each suggested pair; human decisions shall be stored separately from machine predictions. | 3 |
| FR-17 | The user shall search by title/description and filter by class, component, and similarity score; sort by similarity or class. | 4 |
| FR-18 | The system shall show class-distribution and duplicate-result charts. | 4 |
| FR-19 | The system shall export results to CSV. | 4 |
| FR-20 | The trained model and preprocessing shall be saved with Joblib and loaded for prediction; training code shall be separate from app code. | 2 |
| FR-21 | The system shall handle empty descriptions and unseen categories without crashing. | 1-2 |
| FR-22 | Every module shall have unit tests; the whole workflow shall have integration tests. | 1-4 |

## 7. Non-Functional Requirements
- **Usability:** Clear messages; a beginner can run the app with documented commands.
- **Performance:** Target response time for the chosen dataset size **[TO CONFIRM after measurement]**.
- **Reliability:** Invalid input must produce an error message, not a crash.
- **Maintainability:** Separate modules, simple code, fixed random seed, `requirements.txt`.

## 8. Input and Output Formats
**Input CSV:** UTF-8, header row required.
| Column | Required | Notes |
|---|---|---|
| bug_id | Yes | Text, unique |
| title | Yes | Short summary |
| description | Yes | May be empty if the title exists |
| component | No | True label; needed for training |

**Output:** Table and exported CSV with the input columns plus `predicted_component`, `confidence`, `duplicate_of`, `similarity`, `human_decision` (final names **[TO CONFIRM]** in Week 3-4).

## 9. System Workflow
Upload CSV -> Validate -> Preprocess -> Classify -> Find duplicates -> Human review -> Search/filter/charts -> Export CSV.

## 10. Architecture and Modules
| Module | Responsibility |
|---|---|
| `data_loader.py` | Read and validate CSV |
| `preprocessing.py` | Clean and combine text |
| `classifier.py` | Train, evaluate, save, load, predict |
| `duplicate_detector.py` | Similarity search with threshold |
| `evaluation.py` | Metrics and confusion matrix |
| `dataset_prep.py` | Convert the Hugging Face dataset to our CSV format (added) |
| `train_model.py` | Training script: split, compare models, save model (added) |
| `review.py` | Human confirm/reject of duplicate suggestions, kept apart from machine output (added) |
| `reporting.py` | Result tables, charts, CSV export |
| `app.py` | Streamlit interface only (no ML logic) |

## 11. Dataset
**Chosen dataset:** `blak002/Bugzilla_Eclipse_Bug_Reports_Dataset` on Hugging Face. It is a copy of `AliArshad/Bugzilla_Eclipse_Bug_Reports_Dataset`, a processed part of the Lamkanfi, Perez and Demeyer Eclipse/Mozilla defect tracking dataset (MSR 2013). Users must cite that paper.
**Columns seen in the first rows of the source CSV:** Project, Bug ID, Severity Label, Resolution Status, Short Description. There is **no long description** and **no component column**.
**Mapping to our format:** see `src/dataset_prep.py`. Project + Bug ID -> `bug_id`; Short Description -> `title`; `description` stays empty; Severity Label -> `component` (this column holds the class label).
**Consequences:** classification predicts severity from the title only; duplicate detection works on short titles; the dataset has no "duplicate of" links, so duplicate quality cannot be scored automatically.
**[TO CONFIRM]:** licence (the dataset card only asks for citation), exact row count (the Hub viewer lists about 88.7k rows), all values of Severity Label and Resolution Status, class balance, and whether any rows are DUPLICATE.
**Development data:** `data/sample_bug_reports.csv` is a tiny hand-written file for tests only.

## 12. ML and NLP Methodology
- Text = title + description, lowercased and cleaned.
- Features: TF-IDF (fit on training data only to avoid leakage).
- Models: Logistic Regression, Naive Bayes, Linear SVM; best chosen on a validation split, final score on a held-out test set.
- Duplicates: cosine similarity on TF-IDF vectors; optional Sentence Transformers.
- Fixed random seed for reproducibility.

## 13. Testing Strategy and Acceptance Criteria
- Unit tests (Pytest) for every module: valid, empty, missing, invalid, boundary, exceptions.
- Integration tests for each module connection and the full workflow.
- Weekly record: test ID, requirement ID, description, input, expected, actual, pass/fail, bug, fix.
- **Week 1 acceptance:** all data_loader and preprocessing tests pass; loader output feeds preprocessing without errors.

## 14. Risks, Assumptions, Limitations
| Risk | Mitigation |
|---|---|
| Dataset labels are noisy or imbalanced | Inspect first; report per-class metrics |
| Dataset is large and slow on a laptop | Use a subset; record the subset size |
| Embeddings too heavy | Keep TF-IDF baseline as the main method |
| Time overrun | Baseline first, optional features last |
Assumption: you have Python and internet access to download the dataset.

## 15. Four-Week Schedule
| Week | Deliverable |
|---|---|
| 1 | FRD, dataset chosen, `data_loader`, `preprocessing`, tests, integration |
| 2 | Classification pipeline, evaluation, saved model |
| 3 | Duplicate detection, human review, backend integration |
| 4 | Streamlit UI, final testing, documentation, demo |

## 16. Final Deliverables
Source code, Streamlit app, dataset description, trained model, duplicate detector, review workflow, tests, test report, FRD/SRS, requirements.txt, README, screenshots, final report, presentation outline, known limitations.
