"""
Specification tests for the API, written before its implementation (Milestone 4).

Following test-driven development, these tests define the expected endpoints and responses.
They are skipped until ``src/api/main.py`` exists, and will then act as its acceptance criteria.
"""

import pytest

pytestmark = pytest.mark.skip(reason="API implemented in Milestone 4")


@pytest.fixture
def client():
    """Test client of the API application."""
    from fastapi.testclient import TestClient

    from src.api.main import app

    return TestClient(app)


def test_health_check_endpoint(client) -> None:
    """
    The health endpoint confirms that the server is running.
    """
    response = client.get("/health")
    assert response.status_code == 200


def test_predict_spam_endpoint_structure(client) -> None:
    """
    The prediction endpoint returns a label and a confidence score (REQ-F-01).
    """
    payload = {"message": "Congratulations! You won a free iPhone. Click here."}
    response = client.post("/predict", json=payload)
    data = response.json()

    assert response.status_code == 200
    assert data["classification"] in {"spam", "not_spam"}
    assert 0.0 <= data["confidence_score"] <= 1.0


def test_feedback_loop_endpoint(client) -> None:
    """
    The feedback endpoint stores a user correction for later retraining (REQ-F-03).
    """
    payload = {"message_id": "12345", "user_correction": "spam", "summary_useful": False}
    response = client.post("/feedback", json=payload)
    assert response.status_code == 201
