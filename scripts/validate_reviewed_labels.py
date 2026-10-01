"""Validate and normalize human-reviewed NYC 311 department labels."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "evaluation" / "nyc311_real_needs_review.csv"
OUTPUT = ROOT / "data" / "evaluation" / "nyc311_real_reviewed.csv"
VALID_LABELS = {
    "Environment",
    "Social & Health Services",
    "Transport",
    "Non-Complaint",
}


def main() -> None:
    if not INPUT.exists():
        raise FileNotFoundError(f"Review file not found: {INPUT}")
    df = pd.read_csv(INPUT, low_memory=False)
    if "reviewed_department" not in df.columns:
        raise ValueError(
            "Add a reviewed_department column to the review file before validation."
        )
    df["reviewed_department"] = df["reviewed_department"].fillna("").astype(str).str.strip()
    missing = df["reviewed_department"].eq("")
    invalid = ~missing & ~df["reviewed_department"].isin(VALID_LABELS)
    if missing.any() or invalid.any():
        print(f"Rows requiring labels: {int(missing.sum())}")
        if invalid.any():
            print(
                "Invalid labels: "
                + ", ".join(sorted(df.loc[invalid, "reviewed_department"].unique()))
            )
        raise ValueError(
            "Review is incomplete or contains invalid labels. "
            "Use exactly: " + ", ".join(sorted(VALID_LABELS))
        )
    if df["unique_key"].duplicated().any():
        raise ValueError("Duplicate unique_key values found in reviewed data.")
    missing_classes = VALID_LABELS - set(df["reviewed_department"])
    if missing_classes:
        raise ValueError(
            "Final four-class training requires reviewed examples for: "
            + ", ".join(sorted(missing_classes))
        )

    normalized = df.rename(columns={"reviewed_department": "expected_department"})
    normalized["grievance_text"] = (
        normalized[["descriptor", "location_type", "borough"]]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    normalized = normalized[normalized["grievance_text"].ne("")].copy()
    normalized.to_csv(OUTPUT, index=False)
    print(f"validated_rows={len(normalized)}")
    print(normalized["expected_department"].value_counts().to_string())


if __name__ == "__main__":
    main()
