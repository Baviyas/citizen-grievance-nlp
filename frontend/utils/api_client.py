"""
API Client utility for communicating with FastAPI backend
"""

import requests
import streamlit as st
from typing import Optional, Dict, Any
import json

# API Configuration
API_BASE_URL = st.secrets.get("API_BASE_URL", "http://localhost:8000")

class GrievanceAPIClient:
    """Client for interacting with Grievance Management API"""
    
    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url
        self.timeout = 30
    
    def health_check(self) -> bool:
        """Check if API is healthy"""
        try:
            response = requests.get(
                f"{self.base_url}/health",
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
        """Analyze a single grievance"""
        try:
            payload = {
                "description": description,
                "location": location,
                "category": category,
                "contact_email": contact_email,
                "contact_phone": contact_phone
            }
            
            response = requests.post(
                f"{self.base_url}/analyze",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"API Error: {str(e)}")
            return None
    
    def batch_analyze(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Analyze multiple grievances from CSV file"""
        try:
            with open(file_path, "rb") as f:
                files = {"file": f}
                response = requests.post(
                    f"{self.base_url}/batch-analyze",
                    files=files,
                    timeout=self.timeout
                )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"API Error: {str(e)}")
            return None
    
    def get_statistics(self) -> Optional[Dict[str, Any]]:
        """Get system statistics"""
        try:
            response = requests.get(
                f"{self.base_url}/stats",
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"API Error: {str(e)}")
            return None


# Create global client instance
@st.cache_resource
def get_api_client() -> GrievanceAPIClient:
    """Get or create API client (cached)"""
    return GrievanceAPIClient()
