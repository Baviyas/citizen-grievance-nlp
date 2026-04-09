"""
API Client utility for communicating with FastAPI backend
"""

import csv
import os
import uuid
from datetime import datetime
from typing import Optional, Dict, Any, List

import requests
import streamlit as st

# API Configuration
try:
    _api_base = st.secrets.get("API_BASE_URL", None)
except Exception:
    _api_base = None

API_BASE_URL = _api_base or os.getenv("API_BASE_URL", "http://localhost:8000")


def _build_url(base_url: str, path: str) -> str:
    return base_url.rstrip("/") + "/" + path.lstrip("/")


def _map_predict_to_grievance(response: Dict[str, Any], description: str) -> Dict[str, Any]:
    priority = response.get("priority") or response.get("priority_tier") or "P4"
    priority_map = {
        "CRITICAL": "P1",
        "HIGH": "P2",
        "MEDIUM": "P3",
        "LOW": "P4",
        "P1": "P1",
        "P2": "P2",
        "P3": "P3",
        "P4": "P4",
    }
    priority_tier = priority_map.get(str(priority).upper(), "P4")
    sla_map = {
        "P1": "2 hours",
        "P2": "24 hours",
        "P3": "3 days",
        "P4": "7 days",
    }

    urgency_score = response.get("urgency_score", 0.0)
    try:
        urgency_score = float(urgency_score)
    except (ValueError, TypeError):
        urgency_score = 0.0

    if urgency_score <= 10:
        urgency_score = urgency_score * 10

    sentiment_confidence = response.get("sentiment_confidence", 0.0)
    try:
        sentiment_confidence = float(sentiment_confidence)
    except (ValueError, TypeError):
        sentiment_confidence = 0.0

    return {
        "grievance_id": response.get("grievance_id") or f"GRV-{uuid.uuid4().hex[:8].upper()}",
        "description": description,
        "predicted_department": response.get("predicted_department"),
        "confidence": round(float(response.get("department_confidence", response.get("confidence", 0.0))), 4),
        "sentiment": response.get("sentiment", "unknown"),
        "sentiment_score": round(sentiment_confidence * 100 if sentiment_confidence <= 1 else sentiment_confidence, 2),
        "urgency_score": round(urgency_score, 2),
        "priority_tier": priority_tier,
        "sla": sla_map.get(priority_tier, "7 days"),
        "emergency": response.get("emergency", priority_tier == "P1"),
        "timestamp": response.get("timestamp") or datetime.now().isoformat(),
        "recommendations": response.get("recommendations") or [response.get("recommended_action", "")],
    }


def _load_descriptions_from_csv(file_path: str) -> List[str]:
    descriptions = []
    with open(file_path, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if "description" in row and row["description"]:
                descriptions.append(row["description"])
    return descriptions


class GrievanceAPIClient:
    """Client for interacting with Grievance Management API"""

    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url
        self.timeout = 30

    def health_check(self) -> bool:
        """Check if API is healthy"""
        try:
            response = requests.get(
                _build_url(self.base_url, "/health"),
                timeout=self.timeout
            )
            return response.status_code == 200
        except Exception as e:
            st.error(f"API Connection Error: {str(e)}")
            return False

    def analyze_grievance(
        self,
        description: str,
        location: Optional[str] = None,
        category: Optional[str] = None,
        contact_email: Optional[str] = None,
        contact_phone: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Analyze a single grievance using the predict endpoint."""
        payload = {"complaint_text": description}

        try:
            response = requests.post(
                _build_url(self.base_url, "/predict"),
                json=payload,
                timeout=self.timeout
            )

            response.raise_for_status()
            data = response.json()

            # Always map the predict response to grievance format
            return _map_predict_to_grievance(data, description)
        except requests.exceptions.RequestException as e:
            st.error(f"API Error: {str(e)}")
            return None

    def batch_analyze(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Analyze multiple grievances from CSV file using batch_predict endpoint."""
        try:
            descriptions = _load_descriptions_from_csv(file_path)
            if not descriptions:
                st.error("CSV must contain a 'description' column with text.")
                return None

            response = requests.post(
                _build_url(self.base_url, "/batch_predict"),
                json={"complaints": descriptions},
                timeout=self.timeout
            )

            response.raise_for_status()
            data = response.json()

            # Map predictions to grievance format
            mapped_results = [
                _map_predict_to_grievance(pred, desc)
                for pred, desc in zip(data["predictions"], descriptions)
            ]
            return {
                "total_processed": data.get("total_complaints", len(mapped_results)),
                "successful": len(mapped_results),
                "failed": 0,
                "results": mapped_results
            }
        except requests.exceptions.RequestException as e:
            st.error(f"API Error: {str(e)}")
            return None

    def get_statistics(self) -> Optional[Dict[str, Any]]:
        """Get system statistics."""
        try:
            response = requests.get(
                _build_url(self.base_url, "/stats"),
                timeout=self.timeout
            )
            if response.status_code == 404:
                response = requests.get(
                    _build_url(self.base_url, "/metrics"),
                    timeout=self.timeout
                )

            response.raise_for_status()
            data = response.json()

            if "departments" not in data and "department_metrics" in data:
                data = {
                    "departments": list(data.get("department_metrics", {}).keys()),
                    "priority_tiers": ["P1", "P2", "P3", "P4"],
                    "sentiment_scores": list(data.get("sentiment_metrics", {}).keys()),
                    "models": {
                        "routing_model": "unknown",
                        "sentiment_model": "unknown"
                    },
                    "timestamp": data.get("timestamp")
                }

            return data
        except requests.exceptions.RequestException as e:
            st.error(f"API Error: {str(e)}")
            return None


# Create global client instance
@st.cache_resource
def get_api_client() -> GrievanceAPIClient:
    """Get or create API client (cached)"""
    return GrievanceAPIClient()
