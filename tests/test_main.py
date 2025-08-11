# ============================================================================

import pytest
import asyncio
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock

from main import app

class TestHackRxAPI:
    """Test suite for HackRx 6.0 API"""
    
    def setup_method(self):
        """Setup test environment"""
        self.client = TestClient(app)
        
    def test_root_endpoint(self):
        """Test root endpoint returns correct information"""
        response = self.client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "HackRx 6.0" in data["message"]
        assert "endpoints" in data
        
    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = self.client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "components" in data
        
    @pytest.mark.asyncio
    async def test_hackrx_run_endpoint_validation(self):
        """Test request validation"""
        # Test missing fields
        response = self.client.post("/hackrx/run", json={})
        assert response.status_code == 422
        
        # Test invalid URL
        response = self.client.post("/hackrx/run", json={
            "documents": "invalid-url",
            "questions": ["test question"]
        })
        assert response.status_code == 422
        
        # Test empty questions
        response = self.client.post("/hackrx/run", json={
            "documents": "https://example.com/test.pdf",
            "questions": []
        })
        assert response.status_code == 422
        
        # Test too many questions
        response = self.client.post("/hackrx/run", json={
            "documents": "https://example.com/test.pdf",
            "questions": ["question"] * 25
        })
        assert response.status_code == 422

# ============================================================================
