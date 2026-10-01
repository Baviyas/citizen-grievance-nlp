"""Train a provisional three-class model on real NYC 311 records.

The labels come from assistant_review_real_nyc311.py and are not independent
human ground truth. This model is kept separate from the four-class API model.
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
INPUT = ROOT / "data" / "evaluation" / "nyc311_real_assistant_review.csv"
MODEL_DIR = ROOT / "models" / "real_3class"
METRICS = ROOT / "evaluation" / "real_3class_metrics.json"
HOLDOUT = ROOT / "evaluation" / "real_3class_holdout.csv"


def main() -> None:
    df = pd.read_csv(INPUT, low_memory=False)
    required = {
        "unique_key",
        "created_date",
        "grievance_text",
        "assistant_reviewed_department",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df["created_date"] = pd.to_datetime(df["created_date"], errors="coerce")
    df["label"] = df["assistant_reviewed_department"].astype(str)
    df = df.dropna(subset=["created_date"]).drop_duplicates("unique_key")
    df = df[df["grievance_text"].fillna("").str.strip().ne("")].sort_values(
        "created_date"
    )
    labels = sorted(df["label"].unique())
    if labels != ["Environment", "Social & Health Services", "Transport"]:
        raise ValueError(f"Expected three complaint departments, found: {labels}")

    cutoff = df["created_date"].quantile(0.8)
    train = df[df["created_date"] < cutoff].copy()
    test = df[df["created_date"] >= cutoff].copy()
    if train.empty or test.empty:
        raise ValueError("Chronological split produced an empty partition.")

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
                    max_features=20000,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
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
        "dataset": str(INPUT.relative_to(ROOT)),
        "dataset_type": "real NYC 311 records with assistant-reviewed provisional labels",
        "split": "chronological 80/20 by created_date",
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "cutoff": cutoff.isoformat(),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "macro_f1": float(f1_score(y_test, predictions, average="macro")),
        "train_accuracy": float(accuracy_score(y_train, train_predictions)),
        "train_macro_f1": float(
            f1_score(y_train, train_predictions, average="macro")
        ),
        "test_accuracy": float(accuracy_score(y_test, predictions)),
        "test_macro_f1": float(f1_score(y_test, predictions, average="macro")),
        "classes": encoder.classes_.tolist(),
        "classification_report": report,
    }
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_DIR / "pipeline.joblib")
    joblib.dump(encoder, MODEL_DIR / "label_encoder.joblib")
    HOLDOUT.parent.mkdir(parents=True, exist_ok=True)
    test.assign(
        predicted_department=encoder.inverse_transform(predictions),
        confidence=pipeline.predict_proba(test["grievance_text"]).max(axis=1),
    ).to_csv(HOLDOUT, index=False)
    METRICS.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
