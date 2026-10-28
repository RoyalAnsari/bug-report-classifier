"""data_loader.py - Load and validate bug report CSV files.

Purpose:
    Read a CSV file (a file path OR an uploaded file object), check that it
    has the columns we need, clean the basic problems, and return a DataFrame.

Input:  path to a CSV file, or a file-like object (for example a Streamlit upload)
Output: (DataFrame, list of warning messages)
Errors: DataValidationError with a clear message for the user.
"""

import pandas as pd

# Columns every file must have.
REQUIRED_COLUMNS = ["bug_id", "title", "description"]
# Optional column. It holds the true label (component) and is needed for training.
OPTIONAL_COLUMNS = ["component"]


class DataValidationError(Exception):
    """Raised when the uploaded file is not valid. The message is shown to the user."""


def _read_csv(source):
    """Read the CSV and turn pandas errors into clear DataValidationError messages."""
    try:
        # dtype=str keeps everything as text, so ids like "007" are not changed.
        # keep_default_na=False stops pandas from turning words like "NA" into missing values.
        return pd.read_csv(source, dtype=str, keep_default_na=False)
    except pd.errors.EmptyDataError:
        raise DataValidationError("The file is empty.")
    except (pd.errors.ParserError, UnicodeDecodeError):
        raise DataValidationError("The file could not be read as a CSV file.")
    except FileNotFoundError:
        raise DataValidationError("The file was not found.")


def load_bug_reports(source):
    """Load, validate, and lightly clean a bug report CSV.

    Returns:
        (df, warnings) where df has the columns bug_id, title, description
        (plus component if it was present) and warnings is a list of strings.
    """
    df = _read_csv(source)

    # 1. Make column names consistent: " Title " -> "title"
    df.columns = [str(col).strip().lower() for col in df.columns]

    # 2. Check required columns.
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise DataValidationError("Missing required column(s): " + ", ".join(missing))

    # 3. A file with only a header row has no data.
    if df.empty:
        raise DataValidationError("The file has no bug reports (only a header row).")

    # 4. Keep only the columns we know about.
    keep = REQUIRED_COLUMNS + [c for c in OPTIONAL_COLUMNS if c in df.columns]
    df = df[keep].copy()

    # 5. Remove extra spaces from every text value.
    for col in keep:
        df[col] = df[col].str.strip()

    warnings = []

    # 6. Drop rows with no id, or with no text at all (no title and no description).
    no_id = df["bug_id"] == ""
    no_text = (df["title"] == "") & (df["description"] == "")
    bad_rows = no_id | no_text
    if bad_rows.any():
        warnings.append(f"Removed {int(bad_rows.sum())} row(s) with no bug_id or no text.")
        df = df[~bad_rows]

    # 7. Drop repeated bug_id values (keep the first one).
    repeated = df["bug_id"].duplicated()
    if repeated.any():
        warnings.append(f"Removed {int(repeated.sum())} row(s) with a repeated bug_id.")
        df = df[~repeated]

    if df.empty:
        raise DataValidationError("No valid bug reports remain after cleaning.")

    # 8. Warn (but keep) reports that have a title but an empty description.
    empty_desc = int((df["description"] == "").sum())
    if empty_desc:
        warnings.append(f"{empty_desc} report(s) have an empty description (title only).")

    return df.reset_index(drop=True), warnings
