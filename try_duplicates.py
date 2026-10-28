"""Quick manual check of duplicate detection on the real dataset.

Run from the project folder:   python try_duplicates.py
This is NOT part of the app. It only prints a few results so you can look at them.
"""

from src.data_loader import load_bug_reports
from src.duplicate_detector import find_duplicate_pairs
from src.preprocessing import preprocess_reports

SAMPLE_SIZE = 5000   # use a sample so it runs in seconds
THRESHOLD = 0.6     # try 0.6, 0.7, 0.9 and compare

df, _ = load_bug_reports("data/bug_reports.csv")
df = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=42)
clean, _ = preprocess_reports(df)

pairs = find_duplicate_pairs(clean["clean_text"], clean["bug_id"], threshold=THRESHOLD)
print(f"Reports checked: {len(clean)}  |  threshold: {THRESHOLD}  |  pairs found: {len(pairs)}")
pairs = pairs[pairs["similarity"] < 0.99]   # identical titles chhupa do
titles = dict(zip(df["bug_id"], df["title"]))
pairs["query_title"] = pairs["query_id"].map(titles)
pairs["match_title"] = pairs["match_id"].map(titles)
for row in pairs.head(15).itertuples():
    print(f"\n{row.similarity:.2f}  {row.query_id}: {row.query_title}\n      {row.match_id}: {row.match_title}")
