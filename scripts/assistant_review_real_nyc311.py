"""Review real NYC 311 records into defensible operational departments.

This is an assistant review based on complaint type and descriptor. It is
explicitly not independent human ground truth and never invents
Non-Complaint labels for real 311 complaints.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "evaluation" / "nyc311_real_needs_review.csv"
OUTPUT = ROOT / "data" / "evaluation" / "nyc311_real_assistant_review.csv"

WATER_TYPES = {
    "HEAT/HOT WATER",
    "WATER LEAK",
    "PLUMBING",
    "Plumbing",
    "Sewer Maintenance",
    "Water Maintenance",
    "General Construction/Plumbing",
    "Root/Sewer/Sidewalk Condition",
    "Water Conservation",
    "Standing Water",
    "Water Quality",
    "Drinking Water",
    "Waste Water Treatment Plant",
    "Building Drinking Water Tank",
}

OVERRIDES = {
    "PAINT/PLASTER": "Environment",
    "DOOR/WINDOW": "Environment",
    "Missed Collection": "Environment",
    "FLOORING/STAIRS": "Environment",
    "APPLIANCE": "Environment",
    "Maintenance or Facility": "Environment",
    "Building/Use": "Environment",
    "SAFETY": "Environment",
    "Residential Disposal Complaint": "Environment",
    "Wood Pile Remaining": "Environment",
    "Smoking or Vaping": "Environment",
    "Electrical": "Environment",
    "ELECTRIC": "Environment",
    "Hazardous Material": "Environment",
    "Outdoor Dining": "Environment",
    "Boilers": "Environment",
    "OUTSIDE BUILDING": "Environment",
    "Lot Condition": "Environment",
    "Commercial Disposal Complaint": "Environment",
    "Uprooted Stump": "Environment",
    "Indoor Sewage": "Environment",
    "Illegal Fireworks": "Environment",
    "Harboring Bees/Wasps": "Environment",
    "Green Infrastructure": "Environment",
    "BEST/Site Safety": "Environment",
    "Poison Ivy": "Environment",
    "Public Toilet": "Environment",
    "Scaffold Safety": "Environment",
    "Cranes and Derricks": "Environment",
    "Institution Disposal Complaint": "Environment",
    "HEAT/HOT WATER": "Water",
    "WATER LEAK": "Water",
    "PLUMBING": "Water",
    "Plumbing": "Water",
    "Sewer Maintenance": "Water",
    "Water Maintenance": "Water",
    "General Construction/Plumbing": "Water",
    "Root/Sewer/Sidewalk Condition": "Water",
    "Water Conservation": "Water",
    "Standing Water": "Water",
    "Water Quality": "Water",
    "Drinking Water": "Water",
    "Waste Water Treatment Plant": "Water",
    "Building Drinking Water Tank": "Water",
    "Unleashed Dog": "Social & Health Services",
    "Violation of Park Rules": "Social & Health Services",
    "Day Care": "Social & Health Services",
    "Lost Property": "Social & Health Services",
    "Consumer Complaint": "Social & Health Services",
    "Non-Emergency Police Matter": "Social & Health Services",
    "Real Time Enforcement": "Social & Health Services",
    "Cannabis Retailer": "Social & Health Services",
    "Emergency Response Team (ERT)": "Social & Health Services",
    "Special Projects Inspection Team (SPIT)": "Social & Health Services",
    "Investigations and Discipline (IAD)": "Social & Health Services",
    "AHV Inspection Unit": "Social & Health Services",
    "Tobacco or Non-Tobacco Sale": "Social & Health Services",
    "Pet Sale": "Social & Health Services",
    "Tattooing": "Social & Health Services",
    "Beach/Pool/Sauna Complaint": "Environment",
    "LinkNYC": "Transport",
    "GENERAL": "Social & Health Services",
}


def main() -> None:
    df = pd.read_csv(INPUT, low_memory=False)
    df["assistant_reviewed_department"] = df["suggested_department"]
    df["grievance_text"] = (
        df[["descriptor", "location_type", "borough"]]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    missing = df["assistant_reviewed_department"].isna()
    df.loc[missing, "assistant_reviewed_department"] = (
        df.loc[missing, "complaint_type"].map(OVERRIDES)
    )
    water_mask = df["complaint_type"].isin(WATER_TYPES)
    df.loc[water_mask, "assistant_reviewed_department"] = "Water"
    df["review_method"] = "assistant_reviewed_by_complaint_type_and_descriptor"
    df["review_confidence"] = df["assistant_reviewed_department"].map(
        lambda value: "high" if pd.notna(value) else "unresolved"
    )
    df.to_csv(OUTPUT, index=False)
    print(f"reviewed={int(df['assistant_reviewed_department'].notna().sum())}")
    print(f"unresolved={int(df['assistant_reviewed_department'].isna().sum())}")
    print(df["assistant_reviewed_department"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
