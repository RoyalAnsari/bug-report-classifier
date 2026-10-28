# Bug Report Classifier and Duplicate Detector

Upload bug reports (CSV), see their predicted severity, and review possible duplicates.
The machine only suggests duplicates; a person confirms or rejects each one.

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

## Run the app
```
python -m streamlit run app.py
```

## Run the tests
```
python -m pytest -v
```

## Input file format
Columns: `bug_id`, `title`, `description`. Optional: `component`.

(This README is a first version. The final version comes in Week 4.)
