# AI-Driven Citizen Grievance & Sentiment Analysis System

This project is an AI-based system designed to analyze and categorize government grievances using the _NYC 311 Service Requests_ dataset. The pipeline covers everything from raw data ingestion to advanced NLP preprocessing and visual insights.

## Features

* **Data Cleaning:** Automated handling of missing values and date parsing.
* **NLP Pipeline:** Text cleaning, stopword removal (including custom civic terms), and lemmatization.
* **EDA & Insights:** Visualizations of grievance trends, borough-wise heatmaps, and N-gram frequency analysis.
* **Label Mapping:** Consolidated 17+ types into 4 core "Super-Departments" to fix class imbalance.
* **Vectorization:** Converted text to numeric matrices using TF-IDF (Unigrams/Bigrams).
* **Supervised Learning:** Evaluated Logistic Regression and Random Forest via 3-Fold Stratified CV.
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