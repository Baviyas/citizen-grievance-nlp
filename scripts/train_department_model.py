"""Train and evaluate the leakage-safe department routing model."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder


ROOT = Path(__file__).resolve().parents[1]
REVIEWED_FILE = ROOT / "data" / "evaluation" / "nyc311_real_reviewed.csv"
MODEL_DIR = ROOT / "models"

ROUTING_MAP = {
    "Illegal Parking": "Transport",
    "Blocked Driveway": "Transport",
    "Derelict Vehicle": "Transport",
    "Traffic": "Transport",
    "Vending": "Transport",
    "Posting Advertisement": "Transport",
    "Bike/Roller/Skate Chronic": "Transport",
    "Ferry Complaint": "Transport",
    "Noise - Street/Sidewalk": "Environment",
    "Noise - Commercial": "Environment",
    "Noise - Vehicle": "Environment",
    "Noise - Park": "Environment",
    "Noise - House of Worship": "Environment",
    "Graffiti": "Environment",
    "Animal Abuse": "Social & Health Services",
    "Homeless Encampment": "Social & Health Services",
    "Drinking": "Social & Health Services",
    "Panhandling": "Social & Health Services",
    "Disorderly Youth": "Social & Health Services",
    "Urinating in Public": "Social & Health Services",
    "Area Assessment": "Non-Complaint",
    "Community Update": "Non-Complaint",
    "General Feedback": "Non-Complaint",
    "General Inquiry": "Non-Complaint",
    "Information Request": "Non-Complaint",
    "Public Notice": "Non-Complaint",
    "Service Information": "Non-Complaint",
    "Status Check": "Non-Complaint",
}


def build_features(frame: pd.DataFrame) -> pd.Series:
    """Use only information available when a grievance is submitted."""
    columns = ["Descriptor", "Location Type", "Borough"]
    return (
        frame[columns]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def main() -> None:
    if not REVIEWED_FILE.exists():
        raise FileNotFoundError(
            "Human-reviewed labels are required. Run prepare_real_nyc311_labels.py, "
            "fill reviewed_department, then run validate_reviewed_labels.py."
        )
    reviewed = pd.read_csv(REVIEWED_FILE, low_memory=False)
    required = {"unique_key", "grievance_text", "expected_department"}
    missing = required - set(reviewed.columns)
    if missing:
        raise ValueError(f"Reviewed file missing columns: {sorted(missing)}")
    reviewed = reviewed.drop_duplicates("unique_key").copy()
    reviewed["features"] = reviewed["grievance_text"].fillna("").astype(str)
    reviewed["department"] = reviewed["expected_department"].astype(str)

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(
        splitter.split(
            reviewed,
            reviewed["department"],
            groups=reviewed["unique_key"].astype(str),
        )
    )
    train = reviewed.iloc[train_idx]
    test = reviewed.iloc[test_idx]
    holdout_path = ROOT / "evaluation" / "nyc311_reviewed_holdout.csv"
    test.to_csv(holdout_path, index=False)

    encoder = LabelEncoder()
    y_train = encoder.fit_transform(train["department"])
    y_test = encoder.transform(test["department"])

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
                    max_features=10000,
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
    pipeline.fit(train["features"], y_train)
    predictions = pipeline.predict(test["features"])

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
        "feature_columns": ["grievance_text"],
        "split": "GroupShuffleSplit by Unique Key",
        "random_state": 42,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "reviewed_rows": int(len(reviewed)),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "macro_f1": float(f1_score(y_test, predictions, average="macro")),
        "classes": encoder.classes_.tolist(),
        "classification_report": report,
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_DIR / "best_model_pipeline.joblib")
    joblib.dump(encoder, MODEL_DIR / "label_encoder.joblib")
    (MODEL_DIR / "model_meta.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    (ROOT / "evaluation" / "department_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    print(json.dumps({k: metrics[k] for k in ("accuracy", "macro_f1", "classes")}, indent=2))


if __name__ == "__main__":
    main()
