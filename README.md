# AI-Driven Citizen Grievance & Sentiment Analysis System

An end-to-end NLP pipeline that ingests NYC 311 service requests, classifies complaints into municipal departments, scores sentiment, and prioritises tickets by urgency — served through a FastAPI backend and Streamlit frontend.


## Table of Contents

- [Architecture](#architecture)
- [Features](#features)
- [Installation & Setup](#installation--setup)
- [Configuration](#configuration)
- [Priority System](#priority-system)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Deployment](#deployment)
- [Contributors](#contributors)
- [License](#license)

## Architecture

```
┌──────────────────┐     ┌──────────────────┐     ┌───────────────────────┐
│   Frontend       │     │   Backend API    │     │   ML Models           │
│   (Streamlit)    │◄───►│   (FastAPI)      │◄───►│   Transformers        │
│   localhost:8501 │     │   localhost:8000 │     │   + Scikit-learn      │
└──────────────────┘     └──────────────────┘     └───────────────────────┘
         │                        │                          │
         ▼                        ▼                          ▼
  Single / Batch          POST /predict             Sentiment Analysis
  Grievance Input         POST /batch_predict       Department Routing
  CSV Upload              GET  /health               Urgency Scoring
                          GET  /metrics
```

## Features

- **Sentiment Analysis** — 4-class transformer model (positive / neutral / negative / critical) fine-tuned on `distilroberta-base`
- **Department Routing** — TF-IDF + Logistic Regression / Random Forest with 5-fold stratified CV; 17+ complaint types consolidated into 4 super-departments
- **Urgency Scoring** — multi-factor priority score combining sentiment, emergency keywords, model confidence, and recency
- **FastAPI Backend** — production-ready REST API with full endpoint coverage
- **Streamlit Frontend** — web interface for single and batch grievance submission with analytics dashboard
- **Batch Processing** — bulk analysis via CSV upload through `/batch_predict`

## Installation & Setup

### Prerequisites

- Python 3.8+
- pip
- Git

### Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/Baviyas/citizen-grievance-nlp.git
cd citizen-grievance-nlp

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
pip install -r api/requirements.txt
pip install -r frontend/requirements-frontend.txt

# 4. Train models (run notebooks 01- 10 in order, or execute headlessly)

# 5. Start the backend API
cd api && python app.py
# → http://localhost:8000  |  Swagger UI: http://localhost:8000/docs

# 6. Start the frontend (new terminal)
cd frontend && streamlit run app.py
# → http://localhost:8501
```

## Configuration

### Environment Variables

```bash
API_BASE_URL=http://localhost:8000
USE_GPU=1          # 0 for CPU, 1 for GPU
```

### Streamlit Secrets

Create `frontend/.streamlit/secrets.toml`:

```toml
API_BASE_URL = "http://localhost:8000"
USE_GPU = 0
```

## Priority System

| Priority | Score | SLA | Description |
|----------|-------|-----|-------------|
| P1 — Critical | 80 – 100 | 2 hours | Life-threatening / immediate danger |
| P2 — High | 60 – 79 | 24 hours | Urgent infrastructure issues |
| P3 — Medium | 40 – 59 | 3 days | Standard maintenance / repair |
| P4 — Low | 0 – 39 | 7 days | Routine requests |

## Testing

```bash
# Run API test suite
cd api && python -m pytest test_api.py -v

# Manual curl test
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"description": "Water pipe broken, flooding street"}'

# Frontend smoke test
cd frontend && streamlit run app.py
# Visit http://localhost:8501
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Models not found | Run all notebooks in order first |
| Port already in use | `python -m uvicorn api.app:app --port 8001` |
| CUDA out of memory | `export CUDA_VISIBLE_DEVICES=""` to force CPU |
| Import errors | `pip install --upgrade -r requirements.txt` |
| Streamlit secrets error | Create `frontend/.streamlit/secrets.toml` with `API_BASE_URL` |

## Deployment

### Docker

```bash
docker build -t grievance-api .
docker run -p 8000:8000 grievance-api
```

## Contributors

| Name | GitHub |
|------|--------|
| Vasi Khan | [@vasi2904k](https://github.com/vasi2904k) |
| Bhumi Shah | [@code-with-bhumi](https://github.com/code-with-bhumi) |
| Baviya | [@Baviyas](https://github.com/Baviyas) |

## License

This project is licensed under the [MIT License](LICENSE).