# AI-Driven Citizen Grievance & Sentiment Analysis System

This project is an AI-based system designed to analyze and categorize government grievances using the _NYC 311 Service Requests_ dataset. The pipeline covers everything from raw data ingestion to advanced NLP preprocessing and visual insights.

## Features

* **Data Cleaning:** Automated handling of missing values and date parsing.
* **NLP Pipeline:** Text cleaning, stopword removal (including custom civic terms), and lemmatization.
* **EDA & Insights:** Visualizations of grievance trends, borough-wise heatmaps, and N-gram frequency analysis.
* **Label Mapping:** Consolidated 17+ types into 4 core "Super-Departments" to fix class imbalance.
* **Vectorization:** Converted text to numeric matrices using TF-IDF (Unigrams/Bigrams).
* **Supervised Learning:** Evaluated Logistic Regression and Random Forest via 5-Fold Stratified CV.
* **Routing Engine:** Developed an inference helper for real-time, automated complaint routing.
* **Sentiment Analysis:** Added sentiment scoring and urgency classification for enhanced grievance prioritization.

## 📂 Repository Structure

```plaintext
├── data/
│   ├── raw/                 # Original NYC 311 Grievance CSV
│   ├── cleaned/             # Cleaned data
│   └── processed/           # Processed data (with sentiment & urgency)
├── notebooks/
│   ├── 01_data_cleaning.ipynb                    # Structural auditing
│   ├── 02_text_preprocessing.ipynb               # Lemmatization & Tokenization
│   ├── 03_eda_visualizations.ipynb               # Trends, Heatmaps & WordClouds
│   ├── 04_complaint_routing_model.ipynb          # TF-IDF, RF training & CV
│   └── 05_Sentiment_Analysis_and_Urgency_Scoring.ipynb  # Sentiment & urgency analysis
├── models/                  # Saved .pkl files (Pipeline, LabelEncoder, Sentiment models)
├── outputs/                 # Visualization PNGs & Metrics
└── requirements.txt         # Project dependencies and libraries
```

## ⚙️ Installation & Setup

Ensure you have Python installed, then clone the repository and install the necessary dependencies:

1. Clone the repository:
```bash
git clone https://github.com/Baviyas/citizen-grievance-nlp.git
cd citizen-grievance-nlp
```

2. Install Dependencies:

```bash
pip install -r requirements.txt
```

## 🛠️ Project Structure

1. **`01_data_cleaning.ipynb`**: Handles raw data collection and structural cleanup.
2. **`02_text_preprocessing.ipynb`**: Runs the NLP pipeline to prepare text for machine learning.
3. **`03_eda_visualizations.ipynb`**: Generates statistical charts and word clouds.
4. **`04_complaint_routing_model.ipynb`**: Implements TF-IDF vectorization, Supervised Learning (LR & RF), Stratified 3-Fold Cross-Validation, and Inference testing.
5. **`05_Sentiment_Analysis_and_Urgency_Scoring.ipynb`**: Fine-tunes a distil Roberta-base Transformer (RoBERTa) to classify every grievance into 4 sentiment classes — Positive, Neutral, Negative, and Critical/Urgent — and assigns a mathematical urgency score to each complaint.

## 👥 Contributors

| Name | GitHub Account |
| :--- | :--- |
| **Vasi Khan** | https://github.com/vasi2904k |
| **Bhumi Shah** | https://github.com/code-with-bhumi |
| **Baviya** | https://github.com/Baviyas|

