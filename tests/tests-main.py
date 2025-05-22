import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_root_endpoint():
    """Test the root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data
    assert "endpoints" in data

def test_health_check():
    """Test the health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"

def test_docs_endpoint():
    """Test that the docs endpoint is accessible"""
    response = client.get("/docs")
    assert response.status_code == 200

@pytest.mark.asyncio
async def test_document_upload_without_files():
    """Test document upload endpoint without files"""
    response = client.post("/api/documents/upload")
    # Should return an error since no files are provided
    assert response.status_code in [400, 422]

def test_documents_list():
    """Test documents list endpoint"""
    response = client.get("/api/documents/list")
    assert response.status_code == 200
    data = response.json()
    assert "success" in data
    assert "documents" in data

@pytest.mark.asyncio
async def test_research_query_without_documents():
    """Test research query without uploaded documents"""
    query_data = {"query": "What are the main themes?"}
    response = client.post("/api/query/research", json=query_data)
    # Should handle case where no documents are uploaded
    assert response.status_code in [200, 400]