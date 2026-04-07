# Frontend Setup Instructions

Complete frontend package for the Citizen Grievance Management System.

## 📋 What's Included

```
frontend/
├── app.py                    # Main Streamlit application
├── pages_content/            # Page modules
│   ├── home.py
│   ├── submit_grievance.py
│   ├── batch_upload.py
│   ├── analytics.py
│   ├── about.py
│   └── __init__.py
├── utils/                    # Utility modules
│   ├── api_client.py        # API communication
│   ├── ui_components.py     # Reusable UI elements
│   └── __init__.py
├── .streamlit/              # Streamlit configuration
│   ├── config.toml          # Theme and settings
└── └── secrets.toml         # API configuration
```

## Setup Instructions

### Prerequisites
- Python 3.9+
- Virtual environment (venv or conda)
- Backend API running on http://localhost:8000

### Step 1: Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate  # Windows
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Configure API Connection

Edit `.streamlit/secrets.toml`:

```toml
API_BASE_URL = "http://localhost:8000"
USE_GPU = 0
```

Change `API_BASE_URL` if backend is on a different server/port.

### Step 4: Run Frontend

```bash
streamlit run app.py --server.port 8501
```

Or specify a custom port:
```bash
streamlit run app.py --server.port 8502
```

## ⚙️ Configuration

### Streamlit Settings (`.streamlit/config.toml`)

```toml
[theme]
primaryColor = "#667eea"
backgroundColor = "#f0f2f6"
secondaryBackgroundColor = "#e0e1ff"
textColor = "#262730"

[server]
headless = true
port = 8501
enableCORS = false
enableXsrfProtection = true
```

### API Configuration (`.streamlit/secrets.toml`)

```toml
API_BASE_URL = "http://localhost:8000"
USE_GPU = 0
```
## 🌐 Access Application

Open your browser to:
```
http://localhost:8501
```

## 🤝 Support

**Common Tasks:**

**Modify Theme Colors:**
Edit `.streamlit/config.toml` colors:
```toml
[theme]
primaryColor = "#667eea"  # Change this color
```

**Change API Endpoint:**
Edit `.streamlit/secrets.toml`:
```toml
API_BASE_URL = "http://your-backend-server:8000"
```

**Add New Page:**
1. Create new file in `pages_content/`
2. Add function `def show():` 
3. Add to navigation in `app.py`


## 🚀 Next Steps

1. **Setup backend** on your server (handled by backend team)
2. **Deploy frontend** using these instructions
3. **Test integration** between frontend and backend
4. **Customize theme** colors in `.streamlit/config.toml`
5. **Configure API endpoint** in `.streamlit/secrets.toml`
