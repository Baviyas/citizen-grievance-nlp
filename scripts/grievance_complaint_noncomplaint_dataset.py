import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Read the original complaint data
df_complaints = pd.read_csv('./data/raw/grievance_sample.csv')

print(f"Original complaints dataset shape: {df_complaints.shape}")

# Non-complaint templates for different categories
non_complaint_templates = {
    'Noise': [
        "Enjoying the peaceful evening in {location_type}",
        "The neighborhood is quiet and pleasant tonight",
        "Great atmosphere at {location_type} with appropriate sound levels",
        "Peaceful environment in {location_type} area",
        "No disturbances reported in {location_type}"
    ],
    'Parking': [
        "All parking spaces are properly utilized in {location_type}",
        "Vehicles are parked correctly in {location_type}",
        "Good parking availability in {location_type} area",
        "No parking violations observed in {location_type}",
        "Parking situation is well managed in {location_type}"
    ],
    'Driveway': [
        "Driveway access is clear and unobstructed in {location_type}",
        "No issues with driveway accessibility in {location_type}",
        "Driveways are properly maintained in {location_type} area",
        "Clear access to all driveways in {location_type}",
        "Driveway conditions are satisfactory in {location_type}"
    ],
    'General': [
        "Everything is running smoothly in {location_type}",
        "No issues to report in {location_type} area",
        "Conditions are normal and acceptable in {location_type}",
        "Area is well maintained in {location_type}",
        "Satisfactory conditions observed in {location_type}"
    ]
}

# Positive resolution descriptions
positive_resolutions = [
    "The area is well maintained and no action is needed.",
    "Conditions are normal and satisfactory.",
    "No issues were found during the check.",
    "Everything appears to be in good order.",
    "The situation is under control and satisfactory.",
    "No violations or issues detected.",
    "Area is clean and well-maintained.",
    "All systems functioning properly.",
    "No complaints or concerns at this time.",
    "Conditions are within acceptable parameters."
]

def generate_non_complaint_row(original_row):
    """Generate a non-complaint row based on an original complaint row"""
    new_row = original_row.copy()

    # Change complaint type to a neutral category
    complaint_type = original_row.get('Complaint Type', '')
    location_type = original_row.get('Location Type', 'Street/Sidewalk')

    if 'Noise' in complaint_type:
        template_category = 'Noise'
    elif 'Parking' in complaint_type:
        template_category = 'Parking'
    elif 'Driveway' in complaint_type:
        template_category = 'Driveway'
    else:
        template_category = 'General'

    # Generate neutral complaint type and descriptor
    neutral_types = [
        "General Inquiry",
        "Information Request",
        "Status Check",
        "Area Assessment",
        "Community Update",
        "Service Information",
        "Public Notice",
        "General Feedback"
    ]

    neutral_descriptors = [
        "General Information",
        "Area Status",
        "Community Update",
        "Service Check",
        "Public Information",
        "General Inquiry",
        "Status Report",
        "Area Information"
    ]

    new_row['Complaint Type'] = random.choice(neutral_types)
    new_row['Descriptor'] = random.choice(neutral_descriptors)

    # Generate positive resolution description
    new_row['Resolution Description'] = random.choice(positive_resolutions)

    # Add label for classification
    new_row['is_complaint'] = 0  # 0 for non-complaint

    return new_row

# Generate non-complaint data
print("Generating non-complaint data...")
non_complaint_rows = []

for idx, row in df_complaints.iterrows():
    non_complaint_row = generate_non_complaint_row(row)
    non_complaint_rows.append(non_complaint_row)

    if (idx + 1) % 1000 == 0:
        print(f"Generated {idx + 1} non-complaint samples...")

df_non_complaints = pd.DataFrame(non_complaint_rows)

# Add labels to original complaints
df_complaints['is_complaint'] = 1  # 1 for complaint

# Combine datasets
print("Combining datasets...")
df_balanced = pd.concat([df_complaints, df_non_complaints], ignore_index=True)

# Shuffle the dataset
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Balanced dataset shape: {df_balanced.shape}")
print(f"Complaints: {df_balanced['is_complaint'].sum()}")
print(f"Non-complaints: {len(df_balanced) - df_balanced['is_complaint'].sum()}")

# Save the balanced dataset
output_path = './data/raw/grievance_complaint_noncomplaint_dataset.csv'
df_balanced.to_csv(output_path, index=False)

print(f"Balanced dataset saved to: {output_path}")

# Show sample of the balanced data
print("\nSample of balanced dataset:")
print(df_balanced[['Complaint Type', 'Descriptor', 'Resolution Description', 'is_complaint']].head(10))