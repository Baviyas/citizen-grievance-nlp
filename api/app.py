"""
╔════════════════════════════════════════════════════════════════════════════╗
║                                                                            ║
║  FastAPI Application - Citizen Grievance Analysis                         ║
║                                                                            ║
║  Endpoints:                                                               ║
║    POST /predict - Single complaint prediction                            ║
║    POST /batch_predict - Batch prediction                                 ║
║    GET /health - Health check                                             ║
║    GET /metrics - Model metrics                                           ║
║    GET /docs - API documentation (Swagger UI)                            ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝
"""

import os
import json
import torch
import logging
from typing import List, Optional
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import uvicorn

# ════════════════════════════════════════════════════════════════════════════════
# LOGGING CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════════
# REQUEST/RESPONSE SCHEMAS
# ════════════════════════════════════════════════════════════════════════════════

class ComplaintRequest(BaseModel):
    """Schema for single complaint prediction request"""
    complaint_text: str = Field(
        ...,
        description="Raw text of citizen complaint",
        example="Water pipe is broken near my house, no water for 3 days. URGENT!"
    )


class PredictionResponse(BaseModel):
    """Schema for prediction response"""
    complaint_text: str
    predicted_department: str
    department_confidence: float
    sentiment: str
    sentiment_confidence: float
    urgency_score: float
    priority: str
    recommended_action: str
    timestamp: str


class BatchPredictionRequest(BaseModel):
    """Schema for batch prediction"""
    complaints: List[str] = Field(
        ...,
        description="List of complaint texts",
        min_items=1,
        max_items=100
    )


class BatchPredictionResponse(BaseModel):
    """Schema for batch prediction response"""
    total_complaints: int
    predictions: List[PredictionResponse]
    processing_time: float


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    sentiment_model_loaded: bool
    department_model_loaded: bool
    device: str
    timestamp: str


class MetricsResponse(BaseModel):
    """Model metrics response"""
    sentiment_metrics: dict
    department_metrics: dict
    total_predictions: int
    timestamp: str


# ════════════════════════════════════════════════════════════════════════════════
# MODEL MANAGER
# ════════════════════════════════════════════════════════════════════════════════

class ModelManager:
    """Load and manage models"""
    
    def __init__(self, models_dir: str = './models/final_models'):
        self.models_dir = Path(models_dir)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load sentiment model
        try:
            sentiment_path = self.models_dir / 'sentiment_model'
            self.sentiment_tokenizer = AutoTokenizer.from_pretrained(str(sentiment_path))
            self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(
                str(sentiment_path)
            ).to(self.device)
            self.sentiment_model.eval()
            logger.info("✅ Sentiment model loaded")
            self.sentiment_loaded = True
        except Exception as e:
            logger.error(f"❌ Failed to load sentiment model: {e}")
            self.sentiment_loaded = False
        
        # Load department model
        try:
            department_path = self.models_dir / 'department_model'
            self.department_tokenizer = AutoTokenizer.from_pretrained(str(department_path))
            self.department_model = AutoModelForSequenceClassification.from_pretrained(
                str(department_path)
            ).to(self.device)
            self.department_model.eval()
            logger.info("✅ Department model loaded")
            self.department_loaded = True
        except Exception as e:
            logger.error(f"❌ Failed to load department model: {e}")
            self.department_loaded = False
        
        # Load metadata
        try:
            with open(self.models_dir / 'sentiment_metadata.json') as f:
                self.sentiment_metadata = json.load(f)
        except:
            self.sentiment_metadata = {}
        
        try:
            with open(self.models_dir / 'department_metadata.json') as f:
                self.department_metadata = json.load(f)
        except:
            self.department_metadata = {}
    
    def predict_sentiment(self, text: str):
        """Predict sentiment"""
        if not self.sentiment_loaded:
            raise RuntimeError("Sentiment model not loaded")
        
        inputs = self.sentiment_tokenizer(
            text,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.sentiment_model(**inputs)
            logits = outputs.logits
            pred_class = torch.argmax(logits, dim=1).item()
            confidence = torch.softmax(logits, dim=1).max().item()
        
        sentiment_map = {0: 'positive', 1: 'neutral', 2: 'negative', 3: 'critical'}
        return sentiment_map[pred_class], confidence
    
    def predict_department(self, text: str):
        """Predict department"""
        if not self.department_loaded:
            raise RuntimeError("Department model not loaded")
        
        inputs = self.department_tokenizer(
            text,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.department_model(**inputs)
            logits = outputs.logits
            pred_class = torch.argmax(logits, dim=1).item()
            confidence = torch.softmax(logits, dim=1).max().item()
        
        department_map = {
            0: 'water_supply',
            1: 'sanitation',
            2: 'electricity',
            3: 'roads',
            4: 'healthcare',
            5: 'public_safety'
        }
        return department_map[pred_class], confidence


# ════════════════════════════════════════════════════════════════════════════════
# URGENCY & PRIORITY CALCULATOR
# ════════════════════════════════════════════════════════════════════════════════

class UrgencyCalculator:
    """Calculate urgency score and priority level"""
    
    CRITICAL_KEYWORDS = [
        'urgent', 'emergency', 'collapse', 'fire', 'flood', 'danger',
        'explosion', 'leak', 'trapped', 'immediately', 'life risk',
        'critical', 'severe', 'not functioning'
    ]
    
    HIGH_KEYWORDS = [
        'broken', 'not working', 'damaged', 'issue', 'problem',
        'no response', 'poor', 'bad', 'failed', 'blocked',
        'overflowing', 'no access', 'danger'
    ]
    
    @staticmethod
    def calculate_urgency(text: str, sentiment: str, sentiment_confidence: float) -> tuple:
        """
        Calculate urgency score (0-10) and priority level
        Returns: (urgency_score, priority_level)
        """
        text_lower = text.lower()
        
        # Base score from sentiment
        base_scores = {
            'critical': 9.0,
            'negative': 6.0,
            'neutral': 3.0,
            'positive': 1.0
        }
        score = base_scores.get(sentiment, 5.0)
        
        # Adjust for confidence
        score *= sentiment_confidence
        
        # Check for critical keywords
        if any(kw in text_lower for kw in UrgencyCalculator.CRITICAL_KEYWORDS):
            score = max(score, 8.5)
        
        # Check for high keywords
        elif any(kw in text_lower for kw in UrgencyCalculator.HIGH_KEYWORDS):
            score = max(score, 6.0)
        
        # Cap score
        score = min(score, 10.0)
        score = max(score, 0.0)
        
        # Determine priority
        if score >= 8.0:
            priority = 'CRITICAL'
        elif score >= 6.0:
            priority = 'HIGH'
        elif score >= 3.0:
            priority = 'MEDIUM'
        else:
            priority = 'LOW'
        
        return round(score, 2), priority
    
    @staticmethod
    def get_recommended_action(department: str, priority: str) -> str:
        """Get recommended action based on department and priority"""
        actions = {
            'water_supply': {
                'CRITICAL': 'Dispatch emergency team immediately. Restore water supply within 2 hours.',
                'HIGH': 'Schedule repair within 24 hours. Provide alternative water supply.',
                'MEDIUM': 'Create maintenance ticket. Contact resident for survey.',
                'LOW': 'Add to routine maintenance queue. Contact within 7 days.'
            },
            'sanitation': {
                'CRITICAL': 'Dispatch hazmat team immediately. Block affected area.',
                'HIGH': 'Schedule urgent cleanup within 24 hours.',
                'MEDIUM': 'Create service ticket. Schedule within 48 hours.',
                'LOW': 'Add to maintenance queue.'
            },
            'electricity': {
                'CRITICAL': 'Declare emergency. Dispatch team immediately. Safety protocols active.',
                'HIGH': 'Dispatch electrician within 24 hours. Issue power alternatives.',
                'MEDIUM': 'Schedule inspection within 48 hours.',
                'LOW': 'Add to maintenance queue.'
            },
            'roads': {
                'CRITICAL': 'Close affected area. Deploy traffic management.',
                'HIGH': 'Schedule repair within 3 days.',
                'MEDIUM': 'Create work order. Schedule within 7 days.',
                'LOW': 'Add to routine maintenance.'
            },
            'healthcare': {
                'CRITICAL': 'Activate emergency health response protocol.',
                'HIGH': 'Dispatch medical team within 2 hours.',
                'MEDIUM': 'Schedule clinic visit within 48 hours.',
                'LOW': 'Schedule regular appointment.'
            },
            'public_safety': {
                'CRITICAL': 'Alert law enforcement immediately.',
                'HIGH': 'Deploy patrol units within 1 hour.',
                'MEDIUM': 'File formal report. Investigate within 48 hours.',
                'LOW': 'Add to investigation queue.'
            }
        }
        
        return actions.get(department, {}).get(
            priority,
            f"Process complaint with {priority} priority."
        )


# ════════════════════════════════════════════════════════════════════════════════
# FASTAPI APPLICATION
# ════════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="Citizen Grievance Analysis API",
    description="AI-powered API for analyzing citizen complaints and routing to departments",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
model_manager = None
total_predictions = 0
metrics_data = {'sentiment': {}, 'department': {}}


@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    global model_manager, metrics_data
    logger.info("Starting up API server...")
    
    try:
        model_manager = ModelManager()
        
        # Load metrics
        try:
            with open('./evaluation/sentiment_metrics.json') as f:
                metrics_data['sentiment'] = json.load(f)
        except:
            metrics_data['sentiment'] = {}
        
        try:
            with open('./evaluation/department_metrics.json') as f:
                metrics_data['department'] = json.load(f)
        except:
            metrics_data['department'] = {}
        
        logger.info("✅ API ready")
    except Exception as e:
        logger.error(f"❌ Startup error: {e}")


# ════════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ════════════════════════════════════════════════════════════════════════════════

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if model_manager else "unhealthy",
        sentiment_model_loaded=model_manager.sentiment_loaded if model_manager else False,
        department_model_loaded=model_manager.department_loaded if model_manager else False,
        device=model_manager.device if model_manager else "unknown",
        timestamp=datetime.now().isoformat()
    )


@app.get("/metrics", response_model=MetricsResponse, tags=["Metrics"])
async def get_metrics():
    """Get model performance metrics"""
    return MetricsResponse(
        sentiment_metrics=metrics_data.get('sentiment', {}),
        department_metrics=metrics_data.get('department', {}),
        total_predictions=total_predictions,
        timestamp=datetime.now().isoformat()
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_single(request: ComplaintRequest):
    """
    Predict department and urgency for a single complaint
    
    Example request:
    {
        "complaint_text": "Water pipe is broken near my house, no water for 3 days. URGENT!"
    }
    """
    global total_predictions
    
    if not model_manager:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    if not model_manager.sentiment_loaded or not model_manager.department_loaded:
        raise HTTPException(status_code=503, detail="Not all models are loaded")
    
    try:
        # Get predictions
        sentiment, sentiment_conf = model_manager.predict_sentiment(request.complaint_text)
        department, department_conf = model_manager.predict_department(request.complaint_text)
        
        # Calculate urgency and priority
        urgency_score, priority = UrgencyCalculator.calculate_urgency(
            request.complaint_text,
            sentiment,
            sentiment_conf
        )
        
        # Get recommended action
        recommended_action = UrgencyCalculator.get_recommended_action(department, priority)
        
        # Increment counter
        total_predictions += 1
        
        return PredictionResponse(
            complaint_text=request.complaint_text,
            predicted_department=department,
            department_confidence=round(float(department_conf), 4),
            sentiment=sentiment,
            sentiment_confidence=round(float(sentiment_conf), 4),
            urgency_score=urgency_score,
            priority=priority,
            recommended_action=recommended_action,
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch_predict", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(request: BatchPredictionRequest):
    """
    Predict for multiple complaints in batch
    
    Example request:
    {
        "complaints": [
            "Water pipe is broken",
            "Road has huge pothole",
            "Electricity is cut off"
        ]
    }
    """
    global total_predictions
    
    if not model_manager:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    if not model_manager.sentiment_loaded or not model_manager.department_loaded:
        raise HTTPException(status_code=503, detail="Not all models are loaded")
    
    import time
    start_time = time.time()
    
    try:
        predictions = []
        
        for complaint_text in request.complaints:
            # Get predictions
            sentiment, sentiment_conf = model_manager.predict_sentiment(complaint_text)
            department, department_conf = model_manager.predict_department(complaint_text)
            
            # Calculate urgency
            urgency_score, priority = UrgencyCalculator.calculate_urgency(
                complaint_text,
                sentiment,
                sentiment_conf
            )
            
            # Get action
            recommended_action = UrgencyCalculator.get_recommended_action(department, priority)
            
            predictions.append(PredictionResponse(
                complaint_text=complaint_text,
                predicted_department=department,
                department_confidence=round(float(department_conf), 4),
                sentiment=sentiment,
                sentiment_confidence=round(float(sentiment_conf), 4),
                urgency_score=urgency_score,
                priority=priority,
                recommended_action=recommended_action,
                timestamp=datetime.now().isoformat()
            ))
        
        total_predictions += len(request.complaints)
        processing_time = time.time() - start_time
        
        return BatchPredictionResponse(
            total_complaints=len(request.complaints),
            predictions=predictions,
            processing_time=round(processing_time, 2)
        )
    
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Citizen Grievance Analysis API",
        "version": "1.0.0",
        "documentation": "/docs",
        "health_check": "/health",
        "endpoints": {
            "predict": "POST /predict",
            "batch_predict": "POST /batch_predict",
            "metrics": "GET /metrics",
            "health": "GET /health"
        }
    }


# ════════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )