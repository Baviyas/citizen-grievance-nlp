"""Create auditable department labels from real NYC 311 complaint types.

These labels are taxonomy-derived, not human-verified annotations. Ambiguous
types are written to a review file instead of being silently assigned.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "raw" / "nyc311_real_50000.csv"
REVIEW = ROOT / "data" / "evaluation" / "nyc311_real_needs_review.csv"


def classify(value: str) -> str:
    text = str(value).strip().lower()
    if any(
        term in text
        for term in (
            "animal",
            "homeless",
            "encampment",
            "panhandling",
            "drinking",
            "disorderly youth",
            "drug activity",
            "urinating",
        )
    ):
        return "Social & Health Services"
    if any(
        term in text
        for term in (
            "noise",
            "tree",
            "sanitary",
            "unsanitary",
            "dirty",
            "graffiti",
            "air quality",
            "pollution",
            "litter",
            "dump",
            "rodent",
            "sewer",
            "water",
            "waste",
            "recycling",
            "mosquito",
            "asbestos",
            "mold",
            "lead",
            "food",
            "pest",
        )
    ):
        return "Environment"
    if any(
        term in text
        for term in (
            "parking",
            "driveway",
            "traffic",
            "vehicle",
            "street",
            "sidewalk",
            "highway",
            "taxi",
            "ferry",
            "bike",
            "bus stop",
            "scooter",
            "obstruction",
            "curb",
            "bridge",
            "road",
            "sign",
            "vendor",
            "posting",
            "construction",
            "elevator",
        )
    ):
        return "Transport"
    return ""


def main() -> None:
    df = pd.read_csv(INPUT, low_memory=False)
    required = {
        "unique_key",
        "created_date",
        "complaint_type",
        "descriptor",
        "location_type",
        "borough",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df["expected_department"] = df["complaint_type"].map(classify)
    df["grievance_text"] = (
        df[["descriptor", "location_type", "borough"]]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    labeled = df[
        (df["expected_department"] != "") & (df["grievance_text"].str.len() > 0)
    ].copy()
    review = df.copy()
    review["suggested_department"] = review["expected_department"]
    review["reviewed_department"] = ""

    review[
        [
            "unique_key",
            "created_date",
            "complaint_type",
            "descriptor",
            "location_type",
            "borough",
            "suggested_department",
            "reviewed_department",
        ]
    ].to_csv(REVIEW, index=False)

    print(f"labeled={len(labeled)}")
    print(f"needs_review={len(review)}")
    print(labeled["expected_department"].value_counts().to_string())


if __name__ == "__main__":
    main()
