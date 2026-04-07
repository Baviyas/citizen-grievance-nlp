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
│   └── secrets.toml         # API configuration
└── requirements-frontend.txt # Frontend dependencies
```

## 🚀 Setup Instructions

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
pip install -r requirements-frontend.txt
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

## 🌐 Access Application

Open your browser to:
```
http://localhost:8501
```

## 📱 Navigation

The app has 5 main pages accessible from the sidebar:

1. **Home** - Dashboard overview
2. **Submit Grievance** - Single complaint form
3. **Batch Upload** - Process multiple CSV complaints
4. **Analytics** - Statistics and system info
5. **About** - Documentation and system details

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

## 🔧 Troubleshooting

### Issue: "Connection refused" error

**Solution:** Make sure the backend API is running on port 8000
```bash
curl http://localhost:8000/health
```

### Issue: "ModuleNotFoundError"

**Solution:** Ensure all dependencies are installed
```bash
pip install -r requirements-frontend.txt
```

### Issue: Port already in use

**Solution:** Use a different port
```bash
streamlit run app.py --server.port 8502
```

### Issue: API connection timeout

**Solution:** Update `API_BASE_URL` in `.streamlit/secrets.toml` to match your backend URL

## 📦 File Structure

**Main Application Files:**
- `app.py` - Main Streamlit app with navigation menu

**Page Modules:**
- `pages_content/home.py` - Dashboard
- `pages_content/submit_grievance.py` - Single grievance form
- `pages_content/batch_upload.py` - CSV batch processing
- `pages_content/analytics.py` - Statistics
- `pages_content/about.py` - System documentation

**Utility Modules:**
- `utils/api_client.py` - Handles API communication with backend
- `utils/ui_components.py` - Reusable UI components

## 🔌 API Integration

The frontend communicates with the backend API at `http://localhost:8000`

**Main Endpoints Used:**
- `GET /health` - Check backend status
- `POST /analyze` - Analyze single grievance
- `POST /batch-analyze` - Process CSV file
- `GET /stats` - Get system statistics

See backend API documentation for full endpoint details.

## 🧪 Testing

### Test Backend Connection

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{"status":"healthy","models_loaded":true,"timestamp":"..."}
```

### Test Single Grievance Analysis

In the app, go to "Submit Grievance" and submit a test complaint:
- Description: "Pothole on Main Street"
- You should get instant analysis with department routing and priority

### Test Batch Upload

In the app, go to "Batch Upload" and upload a CSV with sample complaints:
```csv
description,location,category
"Pothole on Oak Avenue",Manhattan,
"Water leak in building",Brooklyn,"Social & Health Services"
```

## 📚 Documentation

For more information:
- **Backend README**: See backend repository
- **API Documentation**: Visit `http://localhost:8000/docs` (when backend is running)
- **Architecture Guide**: See ARCHITECTURE.md in project root

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

## 📋 Requirements

**Frontend Dependencies:**
- streamlit>=1.28.0
- streamlit-option-menu>=0.3.2
- plotly>=5.17.0
- requests>=2.31.0
- pandas>=1.5.0
- pytz>=2023.3

**System Requirements:**
- Python 3.9+
- 8GB RAM minimum
- Network access to backend API

## ✅ Verification Checklist

- [ ] Python 3.9+ installed
- [ ] Virtual environment created and activated
- [ ] Dependencies installed: `pip install -r requirements-frontend.txt`
- [ ] Backend API running on localhost:8000
- [ ] `.streamlit/secrets.toml` configured with correct API_BASE_URL
- [ ] Frontend started: `streamlit run app.py`
- [ ] Browser opened to http://localhost:8501
- [ ] Test complaint submitted successfully
- [ ] API health check passes: `curl http://localhost:8000/health`

## 🚀 Next Steps

1. **Setup backend** on your server (handled by backend team)
2. **Deploy frontend** using these instructions
3. **Test integration** between frontend and backend
4. **Customize theme** colors in `.streamlit/config.toml`
5. **Configure API endpoint** in `.streamlit/secrets.toml`
