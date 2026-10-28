# Bug Log

Bugs found during development and how they were fixed.

| ID | Found in | How it was found | Problem | Fix | Test |
|---|---|---|---|---|---|
| BUG-01 | `src/duplicate_detector.py` | Manual check with `try_duplicates.py` on the real dataset (5,000 reports, threshold 0.8) | The default TF-IDF setting ignores one-character words. Version digits such as the "4", "2", "1" in "Bugzilla 4.2.1" were dropped, so "Release notes for Bugzilla 4.2.1" and "... 3.4.8" scored a similarity of 1.00. | Use `token_pattern=r"(?u)\b\w+\b"` so single characters are kept. | TC-DD-22, TC-DD-23 |
| BUG-02 | `src/reporting.py` | Unit test `test_csv_injection_is_blocked` failed | The CSV export protection only looked at columns of type `object`. Newer pandas stores text in a different type, so text starting with `=`, `+`, `-` or `@` was not protected. | Apply the check to every column; only text values are changed. | TC-RP-22 |

## Observations (not bugs)

- Many exact-title pairs on the real data are recurring tasks, such as "Merge fx-sync to mozilla-central". They score 1.00 but are probably separate tasks, not duplicates. This is why a human must confirm every suggestion.
- Some identical titles appear in different projects (for example CDT and Platform). They may be real duplicates. Only a human can decide.
- The dataset has no "duplicate of" links, so duplicate quality cannot be scored automatically (see FRD section 11).
