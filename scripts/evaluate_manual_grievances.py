"""Evaluate the saved department model on manually labeled grievances."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score


ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = ROOT / "data" / "evaluation" / "manual_grievances.csv"
MODEL_FILE = ROOT / "models" / "best_model_pipeline.joblib"
ENCODER_FILE = ROOT / "models" / "label_encoder.joblib"
OUTPUT_FILE = ROOT / "evaluation" / "manual_department_metrics.json"


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Manual holdout file not found: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)
    required = {"grievance_text", "expected_department"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("Manual holdout file is empty")

    pipeline = joblib.load(MODEL_FILE)
    encoder = joblib.load(ENCODER_FILE)
    expected = df["expected_department"].astype(str)
    predicted_ids = pipeline.predict(df["grievance_text"].fillna("").astype(str))
    predicted = pd.Series(
        encoder.inverse_transform(predicted_ids), index=df.index, name="predicted_department"
    )

    report = classification_report(
        expected,
        predicted,
        labels=encoder.classes_.tolist(),
        target_names=encoder.classes_.tolist(),
        output_dict=True,
        zero_division=0,
    )
    df["predicted_department"] = predicted
    df["confidence"] = pipeline.predict_proba(
        df["grievance_text"].fillna("").astype(str)
    ).max(axis=1)
    df.to_csv(ROOT / "evaluation" / "manual_predictions.csv", index=False)

    metrics = {
        "dataset": str(INPUT_FILE.relative_to(ROOT)),
        "dataset_type": "manually authored smoke-test holdout; not production-labeled data",
        "rows": int(len(df)),
        "accuracy": float(accuracy_score(expected, predicted)),
        "macro_f1": float(f1_score(expected, predicted, average="macro")),
        "classes": encoder.classes_.tolist(),
        "classification_report": report,
        "misclassified_rows": int((expected != predicted).sum()),
    }
    OUTPUT_FILE.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
