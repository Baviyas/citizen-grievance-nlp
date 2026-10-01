"""Evaluate the saved model on real NYC 311 taxonomy-labeled data."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "evaluation" / "nyc311_reviewed_holdout.csv"
OUTPUT = ROOT / "evaluation" / "nyc311_real_metrics.json"
PREDICTIONS = ROOT / "evaluation" / "nyc311_real_predictions.csv"


def main() -> None:
    df = pd.read_csv(INPUT)
    pipeline = joblib.load(ROOT / "models" / "best_model_pipeline.joblib")
    encoder = joblib.load(ROOT / "models" / "label_encoder.joblib")
    expected = df["expected_department"].astype(str)
    text = df["grievance_text"].fillna("").astype(str)
    predicted_ids = pipeline.predict(text)
    predicted = encoder.inverse_transform(predicted_ids)

    labels = sorted(expected.unique().tolist())
    report = classification_report(
        expected,
        predicted,
        labels=labels,
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )
    results = df[["unique_key", "created_date", "grievance_text", "complaint_type"]].copy()
    results["expected_department"] = expected
    results["predicted_department"] = predicted
    results["confidence"] = pipeline.predict_proba(text).max(axis=1)
    results.to_csv(PREDICTIONS, index=False)

    metrics = {
        "dataset": str(INPUT.relative_to(ROOT)),
        "dataset_type": "human-reviewed NYC 311 records",
        "rows": int(len(df)),
        "accuracy": float(accuracy_score(expected, predicted)),
        "macro_f1": float(
            f1_score(expected, predicted, labels=labels, average="macro", zero_division=0)
        ),
        "classes_evaluated": labels,
        "classification_report": report,
        "misclassified_rows": int((expected != predicted).sum()),
    }
    OUTPUT.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
