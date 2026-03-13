# AI-Driven Citizen Grievance & Sentiment Analysis System

This project is an AI-based system designed to analyze and categorize government grievances using the _NYC 311 Service Requests_ dataset. The pipeline covers everything from raw data ingestion to advanced NLP preprocessing and visual insights.

## Features

* **Data Cleaning:** Automated handling of missing values and date parsing.
* **NLP Pipeline:** Text cleaning, stopword removal (including custom civic terms), and lemmatization.
* **EDA & Insights:** Visualizations of grievance trends, borough-wise heatmaps, and N-gram frequency analysis.

## Installation

Ensure you have Python installed, then clone the repository and install the necessary dependencies:

```bash
pip install -r requirements.txt

```

## Project Structure

1. **`01_data_cleaning.ipynb`**: Handles raw data collection and structural cleanup.
2. **`02_text_preprocessing.ipynb`**: Runs the NLP pipeline to prepare text for machine learning.
3. **`03_eda_visualizations.ipynb`**: Generates statistical charts and word clouds.

## 📈 Next Steps

Future updates will include **Week 2** modules: TF-IDF vectorization and grievance classification using Naive Bayes and SVM models.
