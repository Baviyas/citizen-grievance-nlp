# AI-Driven Citizen Grievance Analysis System

> Week 4 Deliverable: API Development, Evaluation, and Final Deployment

## Overview

Complete AI-driven system for analyzing citizen grievances and routing them to municipal departments with urgency scoring.

**Components:**
- Sentiment Analysis: 4-class transformer (positive/neutral/negative/critical)
- Department Classification: 6-class routing classifier
- Urgency Scoring: Multi-factor priority calculation
- FastAPI: Production-ready REST API

## Architecture

```
Citizen Complaint Text (Input)
          |
    Preprocessing
          |
   +-----------+   +------------------+
   | Sentiment |   |   Department     |
   | Classifier|   |   Classifier     |
   +-----------+   +------------------+
          |               |
     Urgency Calculator
          |
   Prediction Output
   - Department
   - Sentiment
   - Priority
   - Action
```

## Model Evaluation Results

| Model | Accuracy | F1-Score | Precision | Recall |
|-------|----------|----------|-----------|--------|
| Sentiment | 87.50% | 0.8642 | 87.58% | 87.50% |
| Department | 83.33% | 0.8301 | 83.90% | 83.33% |

## API Endpoints

### POST /predict - Single Complaint Prediction

Request:
```json
{"complaint_text": "Water pipe is broken near my house. URGENT!"}
```

Response:
```json
{
  "predicted_department": "water_supply",
  "department_confidence": 0.9542,
  "sentiment": "critical",
  "sentiment_confidence": 0.9123,
  "urgency_score": 9.25,
  "priority": "CRITICAL",
  "recommended_action": "Dispatch emergency team immediately.",
  "timestamp": "2024-01-15T10:30:45"
}
```

### POST /batch_predict - Bulk Processing (up to 100 complaints)
### GET /health - API and model health status
### GET /metrics - Model performance metrics
### GET /docs - Interactive Swagger UI
### GET /redoc - ReDoc documentation

## Installation & Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt
cd api && pip install -r requirements.txt && cd ..

# 2. Run notebooks in order
jupyter nbconvert --to notebook --execute 08_Model_Evaluation.ipynb
jupyter nbconvert --to notebook --execute 09_Model_Serialization.ipynb
jupyter nbconvert --to notebook --execute 10_FastAPI_Development.ipynb

# 3. Start API server
cd api && python app.py

# 4. Visit: http://localhost:8000/docs
```

## Testing

```bash
cd api
pytest test_api.py -v
# Expected: 22 passed, coverage 95%
```

## Priority Levels

| Priority | Score | Response Time | Example |
|----------|-------|---------------|----------|
| CRITICAL | 8-10 | Immediate (30 min) | Fire, flood, trapped |
| HIGH | 6-7.9 | 24 hours | Broken pipe, no power |
| MEDIUM | 3-5.9 | 3 days | Minor damage, repair |
| LOW | 0-2.9 | 7 days | Routine maintenance |

## Project Structure

```
citizen-grievance-analysis/
|-- 08_Model_Evaluation.ipynb
|-- 09_Model_Serialization.ipynb
|-- 10_FastAPI_Development.ipynb
|-- 11_Documentation_and_CICD.ipynb
|-- data/processed/grievance_processed.csv
|-- models/final_models/
|   |-- sentiment_model/
|   |-- sentiment_metadata.json
|   |-- department_model/
|   `-- department_metadata.json
|-- evaluation/
|   |-- sentiment_metrics.json
|   |-- sentiment_cm.png
|   |-- department_metrics.json
|   `-- department_cm.png
|-- api/
|   |-- app.py
|   |-- test_api.py
|   `-- requirements.txt
|-- .github/
|   |-- ARCHITECTURE_DECISIONS.md
|   `-- ci_cd.yml
|-- README.md
|-- GIT_COMMITS.md
`-- requirements.txt
```

## Troubleshooting

```bash
# Models not found -> run notebooks 08 and 09 first
# Port in use -> python -m uvicorn api.app:app --port 8001
# Out of memory -> export CUDA_VISIBLE_DEVICES="" (force CPU)
# Import errors -> pip install --upgrade -r requirements.txt
```

---
Version: 1.0.0 | Status: Production Ready | Last Updated: 2024-01-15
