# Real-data label review

The final department model must use human-reviewed labels. The downloaded NYC
311 data contains real submissions, but it does not contain the project's four
department labels. The taxonomy suggestion is therefore only an annotation
aid, not ground truth.

## Workflow

```powershell
.venv\Scripts\python.exe scripts\prepare_real_nyc311_labels.py
```

Open `data/evaluation/nyc311_real_needs_review.csv` and fill the
`reviewed_department` column using exactly one of:

- `Environment`
- `Non-Complaint`
- `Social & Health Services`
- `Transport`

Do not copy `suggested_department` without checking the grievance text and
complaint type. Add genuinely reviewed `Non-Complaint` examples from a
separate source because ordinary NYC 311 records are complaints.

Then validate and train:

```powershell
.venv\Scripts\python.exe scripts\validate_reviewed_labels.py
.venv\Scripts\python.exe scripts\train_department_model.py
.venv\Scripts\python.exe scripts\evaluate_real_nyc311.py
```

Validation refuses incomplete labels, duplicate request IDs, invalid category
names, or missing classes. Until this review is completed, existing model
metrics are provisional and must not be presented as a final production
benchmark.

For an assistant-generated operational review (not human ground truth), run:

```powershell
.venv\Scripts\python.exe scripts\assistant_review_real_nyc311.py
```

This produces `data/evaluation/nyc311_real_assistant_review.csv`. It labels
real 311 complaints into the three defensible complaint departments and does
not fabricate `Non-Complaint` labels.
