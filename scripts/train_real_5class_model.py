"""Train the production routing model with a curated Non-Complaint class.

The four complaint classes come from real NYC 311 records with assistant
review. Non-Complaint examples are explicitly authored neutral/service
requests and are documented as curated data, not real 311 complaints.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[1]
COMPLAINTS = ROOT / "data" / "evaluation" / "nyc311_real_assistant_review.csv"
NON_COMPLAINTS = ROOT / "data" / "evaluation" / "manual_training_grievances.csv"
MODEL_DIR = ROOT / "models" / "real_5class"
METRICS = ROOT / "evaluation" / "real_5class_metrics.json"
HOLDOUT = ROOT / "evaluation" / "real_5class_holdout.csv"
CLASSES = [
    "Environment",
    "Non-Complaint",
    "Social & Health Services",
    "Transport",
    "Water",
]


def main() -> None:
    complaints = pd.read_csv(COMPLAINTS, low_memory=False)
    required = {
        "unique_key",
        "created_date",
        "grievance_text",
        "assistant_reviewed_department",
    }
    missing = required - set(complaints.columns)
    if missing:
        raise ValueError(f"Complaint data missing columns: {sorted(missing)}")
    complaints["created_date"] = pd.to_datetime(
        complaints["created_date"], errors="coerce"
    )
    complaints = complaints.dropna(subset=["created_date"]).drop_duplicates("unique_key")
    complaints = complaints[
        complaints["grievance_text"].fillna("").str.strip().ne("")
    ].copy()
    complaints["label"] = complaints["assistant_reviewed_department"].astype(str)

    curated = pd.read_csv(NON_COMPLAINTS, low_memory=False)
    curated = curated[curated["expected_department"].eq("Non-Complaint")].copy()
    if len(curated) < 8:
        raise ValueError("At least 8 curated Non-Complaint examples are required.")
    curated["created_date"] = pd.Timestamp("2025-01-01")
    curated["unique_key"] = ["curated-non-complaint-" + str(i) for i in curated.index]
    curated["grievance_text"] = curated["grievance_text"].fillna("").astype(str)
    curated["label"] = "Non-Complaint"

    complaints["source"] = "real_nyc311_assistant_review"
    curated["source"] = "curated_neutral_examples"
    data = pd.concat(
        [
            complaints[
                ["unique_key", "created_date", "grievance_text", "label", "source"]
            ],
            curated[
                ["unique_key", "created_date", "grievance_text", "label", "source"]
            ],
        ],
        ignore_index=True,
    ).sort_values("created_date")

    # Keep a small curated Non-Complaint holdout so every class is evaluated.
    real_cutoff = complaints["created_date"].quantile(0.8)
    real_train = complaints[complaints["created_date"] < real_cutoff]
    real_test = complaints[complaints["created_date"] >= real_cutoff]
    curated = curated.sort_values("unique_key")
    curated_cutoff = max(1, int(len(curated) * 0.8))
    curated_train = curated.iloc[:curated_cutoff]
    curated_test = curated.iloc[curated_cutoff:]
    train = pd.concat([real_train, curated_train], ignore_index=True)
    test = pd.concat([real_test, curated_test], ignore_index=True)

    encoder = LabelEncoder()
    y_train = encoder.fit_transform(train["label"])
    y_test = encoder.transform(test["label"])
    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95,
                    sublinear_tf=True,
                    strip_accents="unicode",
                    max_features=25000,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2500,
                    class_weight="balanced",
                    C=1.0,
                    solver="liblinear",
                    random_state=42,
                ),
            ),
        ]
    )
    pipeline.fit(train["grievance_text"], y_train)
    train_predictions = pipeline.predict(train["grievance_text"])
    predictions = pipeline.predict(test["grievance_text"])
    report = classification_report(
        y_test,
        predictions,
        labels=list(range(len(encoder.classes_))),
        target_names=encoder.classes_,
        output_dict=True,
        zero_division=0,
    )
    metrics = {
        "model": "TF-IDF + Logistic Regression",
        "dataset_type": "real NYC complaints plus curated Non-Complaint examples",
        "split": "chronological 80/20 for real complaints; 80/20 holdout for curated Non-Complaint examples",
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "curated_non_complaint_rows": int(len(curated)),
        "cutoff": real_cutoff.isoformat(),
        "train_accuracy": float(accuracy_score(y_train, train_predictions)),
        "test_accuracy": float(accuracy_score(y_test, predictions)),
        "train_macro_f1": float(f1_score(y_train, train_predictions, average="macro")),
        "test_macro_f1": float(f1_score(y_test, predictions, average="macro")),
        "classes": encoder.classes_.tolist(),
        "classification_report": report,
    }
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_DIR / "pipeline.joblib")
    joblib.dump(encoder, MODEL_DIR / "label_encoder.joblib")
    test.assign(
        predicted_department=encoder.inverse_transform(predictions),
        confidence=pipeline.predict_proba(test["grievance_text"]).max(axis=1),
    ).to_csv(HOLDOUT, index=False)
    METRICS.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
