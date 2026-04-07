"""
Comprehensive test suite for Citizen Grievance Analysis API
"""
import pytest
import json
from fastapi.testclient import TestClient
from app import app, UrgencyCalculator, model_manager


@pytest.fixture(scope="session", autouse=True)
def initialize_models():
    """Initialize models before running tests"""
    from app import model_manager
    if model_manager is None:
        from app import ModelManager
        # Override the global model_manager
        import app
        app.model_manager = ModelManager()
    yield


client = TestClient(app)

class TestHealthEndpoint:
    """Test health check endpoint"""
    
    def test_health_check(self):
        """Test health endpoint returns 200"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["healthy", "unhealthy"]
        assert "device" in data
        assert "timestamp" in data

    def test_health_check_has_model_status(self):
        """Test health includes model status"""
        response = client.get("/health")
        data = response.json()
        assert "sentiment_model_loaded" in data
        assert "department_model_loaded" in data

class TestMetricsEndpoint:
    """Test metrics endpoint"""
    def test_metrics_endpoint(self):
        """Test metrics endpoint returns 200"""
        response = client.get("/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "sentiment_metrics" in data
        assert "department_metrics" in data
        assert "total_predictions" in data
        assert "timestamp" in data

class TestPredictEndpoint:
    """Test single prediction endpoint"""
    
    def test_predict_valid_request(self):
        """Test valid prediction request"""
        payload = {
            "complaint_text": "Water pipe is broken near my house, no water for 3 days."
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure
        assert "complaint_text" in data
        assert "predicted_department" in data
        assert "department_confidence" in data
        assert "sentiment" in data
        assert "sentiment_confidence" in data
        assert "urgency_score" in data
        assert "priority" in data
        assert "recommended_action" in data
        assert "timestamp" in data
    
    def test_predict_various_complaints(self):
        """Test predictions on various complaints"""
        test_cases = [
            "Road has a large pothole blocking traffic",
            "Hospital is not providing proper treatment",
            "Electricity is cut off in my area",
            "Toilet overflow causing health hazard",
            "Thank you for fixing the water issue"
        ] 
        for complaint in test_cases:
            response = client.post("/predict", json={"complaint_text": complaint})
            assert response.status_code == 200
            data = response.json()
            assert data["predicted_department"] in [
                "water_supply", "sanitation", "electricity", "roads",
                "healthcare", "public_safety"
            ]
            assert data["sentiment"] in ["positive", "neutral", "negative", "critical"]
            assert 0 <= data["urgency_score"] <= 10
            assert data["priority"] in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    
    def test_predict_empty_text(self):
        """Test prediction with empty text"""
        payload = {"complaint_text": ""}
        response = client.post("/predict", json=payload)
        # Should still process (may return default classification)
        assert response.status_code in [200, 422]

    def test_predict_long_text(self):
        """Test prediction with very long text"""
        long_text = "complaint text " * 100
        payload = {"complaint_text": long_text}
        response = client.post("/predict", json=payload)
        # Should handle through truncation
        assert response.status_code == 200


class TestBatchPredictEndpoint:
    """Test batch prediction endpoint"""
    
    def test_batch_predict_valid_request(self):
        """Test valid batch prediction"""
        payload = {
            "complaints": [
                "Water pipe is broken",
                "Road has pothole",
                "Electricity cut off"
            ]
        }
        response = client.post("/batch_predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        
        assert data["total_complaints"] == 3
        assert len(data["predictions"]) == 3
        assert "processing_time" in data
        
        # Verify each prediction
        for pred in data["predictions"]:
            assert "predicted_department" in pred
            assert "sentiment" in pred
            assert "urgency_score" in pred
    
    def test_batch_predict_single_complaint(self):
        """Test batch with single complaint"""
        payload = {"complaints": ["Water issue"]}
        response = client.post("/batch_predict", json=payload)
        assert response.status_code == 200
        assert response.json()["total_complaints"] == 1
    
    def test_batch_predict_max_complaints(self):
        """Test batch prediction with max complaints"""
        complaints = ["complaint " + str(i) for i in range(50)]
        payload = {"complaints": complaints}
        response = client.post("/batch_predict", json=payload)
        assert response.status_code == 200
        assert response.json()["total_complaints"] == 50

    def test_batch_predict_exceeds_max(self):
        """Test batch with too many complaints"""
        complaints = ["complaint " + str(i) for i in range(101)]
        payload = {"complaints": complaints}
        response = client.post("/batch_predict", json=payload)
        # Should reject if exceeds max_items
        assert response.status_code == 422

class TestUrgencyCalculator:
    """Test urgency calculation logic"""
    
    def test_critical_priority_assignment(self):
        """Test critical priority for high urgency"""
        text = "URGENT! Water pipe collapsed - emergency!"
        urgency, priority = UrgencyCalculator.calculate_urgency(
            text, "critical", 0.95
        )
        assert priority == "CRITICAL"
        assert urgency >= 8.0
    
    def test_high_priority_assignment(self):
        """Test HIGH priority assignment"""
        text = "Pipe is broken, no water"
        urgency, priority = UrgencyCalculator.calculate_urgency(
            text, "negative", 0.85
        )
        assert priority in ["HIGH", "CRITICAL"]
    
    def test_low_priority_assignment(self):
        """Test LOW priority assignment"""
        text = "All good, thank you"
        urgency, priority = UrgencyCalculator.calculate_urgency(
            text, "positive", 0.90
        )
        assert priority == "LOW"
        assert urgency <= 5.0

    def test_urgency_bounds(self):
        """Test urgency score stays within bounds"""
        test_texts = [
            ("EMERGENCY!!!", "critical", 0.95),
            ("No issues", "positive", 0.90),
            ("Minor problem", "neutral", 0.70)
        ]
        
        for text, sentiment, conf in test_texts:
            urgency, _ = UrgencyCalculator.calculate_urgency(text, sentiment, conf)
            assert 0 <= urgency <= 10

class TestRootEndpoint:
    """Test root endpoint"""
    def test_root_endpoint(self):
        """Test root endpoint returns API info"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "endpoints" in data

class TestRequestValidation:
    """Test request validation"""
    def test_missing_complaint_text(self):
        """Test missing complaint_text field"""
        response = client.post("/predict", json={})
        assert response.status_code == 422

    def test_invalid_json(self):
        """Test invalid JSON"""
        response = client.post(
            "/predict",
            content="invalid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code in [400, 422]

class TestErrorHandling:
    """Test error handling"""
    def test_404_endpoint(self):
        """Test 404 for non-existent endpoint"""
        response = client.get("/nonexistent")
        assert response.status_code == 404


# ════════════════════════════════════════════════════════════════════════════════
# RUN TESTS
# ════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Run: pytest test_api.py -v
    pytest.main([__file__, "-v", "--tb=short"])