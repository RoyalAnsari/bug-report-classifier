"""dataset_prep.py - Convert the Hugging Face Bugzilla/Eclipse dataset to our CSV format.

The public dataset has these columns:
    Project, Bug ID, Severity Label, Resolution Status, Short Description
Our project expects:
    bug_id, title, description, component

Mapping:
    Project + Bug ID  -> bug_id        (Project is added because different projects may reuse numbers)
    Short Description -> title
    (none)            -> description   (the dataset has no long description, so it is left empty)
    label column      -> component     (we use "Severity Label" by default; it is our class label)

Run on your computer (needs internet and the "datasets" package):
    python -m src.dataset_prep
"""

import pandas as pd

HF_DATASET_NAME = "blak002/Bugzilla_Eclipse_Bug_Reports_Dataset"
SOURCE_COLUMNS = ["Project", "Bug ID", "Short Description"]
DEFAULT_LABEL_COLUMN = "Severity Label"
OUTPUT_PATH = "data/bug_reports.csv"


def prepare_bug_reports(raw_df, label_column=DEFAULT_LABEL_COLUMN):
    """Return a DataFrame with the columns bug_id, title, description, component."""
    needed = SOURCE_COLUMNS + [label_column]
    missing = [col for col in needed if col not in raw_df.columns]
    if missing:
        raise ValueError(
            "Dataset is missing column(s): " + ", ".join(missing)
            + ". Columns found: " + ", ".join(map(str, raw_df.columns))
        )

    project = raw_df["Project"].fillna("").astype(str).str.strip()
    bug_number = raw_df["Bug ID"].fillna("").astype(str).str.strip()
    # A missing number must stay empty so data_loader can remove the row.
    bug_id = (project + "-" + bug_number).where(bug_number != "", "")

    return pd.DataFrame(
        {
            "bug_id": bug_id,
            "title": raw_df["Short Description"].fillna("").astype(str).str.strip(),
            "description": "",
            "component": raw_df[label_column].fillna("").astype(str).str.strip(),
        }
    )


def main():
    """Download the dataset, convert it, and save it as data/bug_reports.csv."""
    from datasets import load_dataset  # imported here so tests do not need it

    ds = load_dataset(HF_DATASET_NAME)
    raw_df = ds["train"].to_pandas()
    print("Columns found:", list(raw_df.columns))
    result = prepare_bug_reports(raw_df)
    result.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(result)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
