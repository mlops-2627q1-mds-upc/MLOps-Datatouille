"""
Unit tests for the API endpoints, run through FastAPI's ``TestClient``.

The production model is replaced by a small classifier trained in ``conftest.py`` and the
Gemini client of the summarizer is mocked, so the tests need neither network access nor
credentials. The prediction database is created in a temporary directory.
"""

from collections.abc import Iterator
from pathlib import Path
import sqlite3
from unittest.mock import MagicMock

from fastapi.testclient import TestClient
import pytest
from sklearn.pipeline import Pipeline

from src.api import api
from src.api.model_loader import LoadedModel, ModelLoadError
from src.api.settings import API_VERSION, MAX_BATCH_SIZE, MAX_TEXT_LENGTH
from src.api.summarizer import Summarizer

SPAM_MESSAGE = "WIN a FREE prize now!!! Claim your cash reward at http://win.example"
NOT_SPAM_MESSAGE = "Hi team, the project meeting has been moved to Monday at 10."
CONTACT_MESSAGE = "Thanks for the update. Write to ana@example.com or call +34 612 345 678."
TEST_MODEL_VERSION = "7"
TEST_SUMMARY = "The meeting has been moved to Monday."


@pytest.fixture(scope="module")
def db_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Temporary SQLite file of predictions and feedback, shared by the module."""
    return tmp_path_factory.mktemp("api") / "predictions.db"


@pytest.fixture(scope="module")
def gemini() -> MagicMock:
    """Mocked Gemini client of the summarizer, shared by the module."""
    return MagicMock()


@pytest.fixture(autouse=True)
def reset_gemini(gemini: MagicMock) -> None:
    """Reset the mocked Gemini client before each test, answering with ``TEST_SUMMARY``."""
    gemini.reset_mock(return_value=True, side_effect=True)
    gemini.models.generate_content.return_value.text = TEST_SUMMARY


@pytest.fixture(scope="module")
def client(spam_pipeline: Pipeline, db_path: Path, gemini: MagicMock) -> Iterator[TestClient]:
    """
    Test client with the lifespan run, serving the test classifier.

    The ``with`` block runs the startup and shutdown of the application. Server errors are
    returned as responses instead of raised, as a real client would receive them.
    """
    loaded = LoadedModel(
        pipeline=spam_pipeline,
        name="spam-classifier",
        alias="production",
        version=TEST_MODEL_VERSION,
        run_id="test-run",
        metrics={"cv_f1_macro": 0.99, "test_f1_macro": 0.98},
    )
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(api, "load_production_model", lambda name, alias: loaded)
        patch.setattr(api, "PREDICTIONS_DB_PATH", db_path)
        summarizer = Summarizer(gemini, "test-model")
        patch.setattr(api, "build_summarizer", lambda api_key, model, timeout: summarizer)
        with TestClient(api.app, raise_server_exceptions=False) as test_client:
            yield test_client


def predict(client: TestClient, texts: list[str], **extra: object) -> dict:
    """Call /predict and return the JSON body of a successful response."""
    response = client.post("/predict", json={"texts": texts, **extra})
    assert response.status_code == 200, response.text
    return response.json()


def test_root_returns_name_and_version(client: TestClient) -> None:
    """Test of the welcome message with the API name and version."""
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert body["version"] == API_VERSION
    assert body["name"]
    assert "/docs" in body["message"]


def test_health_reports_loaded_model(client: TestClient) -> None:
    """Test of the health check with the loaded model name and version."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_loaded": True,
        "model_name": "spam-classifier",
        "model_version": TEST_MODEL_VERSION,
    }


def test_model_returns_metadata_and_metrics(client: TestClient) -> None:
    """Test of the metadata of the production model."""
    response = client.get("/model")
    assert response.status_code == 200
    body = response.json()
    assert body["alias"] == "production"
    assert body["version"] == TEST_MODEL_VERSION
    assert body["metrics"]["cv_f1_macro"] == pytest.approx(0.99)


def test_predict_single_message(client: TestClient) -> None:
    """Test of a prediction for one message, with scores in [0, 1] and the model version."""
    body = predict(client, [SPAM_MESSAGE])
    assert body["model_version"] == TEST_MODEL_VERSION
    assert body["threshold"] == 0.5
    [result] = body["predictions"]
    assert result["label"] in {"spam", "not_spam"}
    assert 0.0 <= result["confidence"] <= 1.0
    assert 0.0 <= result["spam_probability"] <= 1.0
    assert result["prediction_id"] >= 1


def test_predict_returns_no_summary_and_does_not_call_gemini(
    client: TestClient, gemini: MagicMock
) -> None:
    """Test of the deprecated summary field, always null, with no Gemini request."""
    body = predict(client, [SPAM_MESSAGE, NOT_SPAM_MESSAGE])
    assert all(result["summary"] is None for result in body["predictions"])
    gemini.models.generate_content.assert_not_called()


def test_summarize_returns_summary(client: TestClient) -> None:
    """Test of a summary returned by the summarizer."""
    response = client.post("/summarize", json={"text": NOT_SPAM_MESSAGE})
    assert response.status_code == 200
    assert response.json() == {"summary": TEST_SUMMARY}


def test_summarize_sends_anonymized_message(client: TestClient, gemini: MagicMock) -> None:
    """Test of the anonymization of contact details before the message is sent to Gemini."""
    assert client.post("/summarize", json={"text": CONTACT_MESSAGE}).status_code == 200
    contents = gemini.models.generate_content.call_args.kwargs["contents"]
    assert "xxemail" in contents and "xxphone" in contents
    assert "ana@example.com" not in contents and "612" not in contents


def test_summarize_failure_returns_502_without_details(
    client: TestClient, gemini: MagicMock
) -> None:
    """Test of the 502 response when Gemini fails, which hides the error from the client."""
    gemini.models.generate_content.side_effect = RuntimeError("secret internal detail")
    response = client.post("/summarize", json={"text": NOT_SPAM_MESSAGE})
    assert response.status_code == 502
    assert response.json() == {"detail": "Summary service unavailable"}


def test_summarize_returns_503_without_summarizer(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test of the 503 response when summarization is not configured."""
    monkeypatch.setattr(api.app.state, "summarizer", None)
    response = client.post("/summarize", json={"text": NOT_SPAM_MESSAGE})
    assert response.status_code == 503
    assert client.post("/predict", json={"texts": [NOT_SPAM_MESSAGE]}).status_code == 200


@pytest.mark.parametrize(
    "payload",
    [{"text": ""}, {"text": "   "}, {"text": "a" * (MAX_TEXT_LENGTH + 1)}, {}],
    ids=["empty", "blank", "too-long", "missing"],
)
def test_summarize_rejects_invalid_input(client: TestClient, payload: dict) -> None:
    """Test of the 422 response for summary requests that do not match the schema."""
    assert client.post("/summarize", json=payload).status_code == 422


def test_predict_batch_keeps_order_and_labels_known_examples(client: TestClient) -> None:
    """Test of a batch prediction labelling known spam and not spam messages in order."""
    body = predict(client, [SPAM_MESSAGE, NOT_SPAM_MESSAGE])
    labels = [result["label"] for result in body["predictions"]]
    assert labels == ["spam", "not_spam"]
    ids = [result["prediction_id"] for result in body["predictions"]]
    assert len(set(ids)) == 2


def test_predict_confidence_is_probability_of_returned_label(client: TestClient) -> None:
    """Test of the confidence as the probability of the returned label."""
    for result in predict(client, [SPAM_MESSAGE, NOT_SPAM_MESSAGE])["predictions"]:
        expected = result["spam_probability"]
        if result["label"] == "not_spam":
            expected = 1.0 - expected
        assert result["confidence"] == pytest.approx(expected)


def test_predict_threshold_sets_decision_boundary(client: TestClient) -> None:
    """Test of the threshold: a message is spam when its spam probability reaches it."""
    probability = predict(client, [SPAM_MESSAGE])["predictions"][0]["spam_probability"]
    assert probability < 1.0
    at_boundary = predict(client, [SPAM_MESSAGE], threshold=probability)["predictions"][0]
    assert at_boundary["label"] == "spam"
    above = predict(client, [SPAM_MESSAGE], threshold=(probability + 1.0) / 2)["predictions"][0]
    assert above["label"] == "not_spam"
    assert above["confidence"] == pytest.approx(1.0 - probability)


def test_predict_stores_anonymized_message(client: TestClient, db_path: Path) -> None:
    """Test of the storage of each prediction with contact details anonymized."""
    prediction_id = predict(client, [CONTACT_MESSAGE])["predictions"][0]["prediction_id"]
    with sqlite3.connect(db_path) as connection:
        (text,) = connection.execute(
            "SELECT text FROM predictions WHERE id = ?", (prediction_id,)
        ).fetchone()
    assert "xxemail" in text and "xxphone" in text
    assert "ana@example.com" not in text and "612" not in text


@pytest.mark.parametrize(
    "payload",
    [
        {"texts": []},
        {"texts": ["   "]},
        {"texts": ["a" * (MAX_TEXT_LENGTH + 1)]},
        {"texts": ["hello"] * (MAX_BATCH_SIZE + 1)},
        {"texts": ["hello"], "threshold": 1.5},
        {"texts": "hello"},
        {},
    ],
    ids=["empty", "blank", "too-long", "too-many", "threshold", "not-a-list", "missing"],
)
def test_predict_rejects_invalid_input(client: TestClient, payload: dict) -> None:
    """Test of the 422 response for requests that do not match the schema."""
    assert client.post("/predict", json=payload).status_code == 422


def test_feedback_stores_correction(client: TestClient, db_path: Path) -> None:
    """Test of a stored correction linked to its prediction and model version."""
    prediction = predict(client, [NOT_SPAM_MESSAGE])["predictions"][0]
    response = client.post(
        "/feedback",
        json={"prediction_id": prediction["prediction_id"], "corrected_label": "spam"},
    )
    assert response.status_code == 201
    body = response.json()
    assert body["prediction_id"] == prediction["prediction_id"]
    assert body["predicted_label"] == prediction["label"]
    assert body["corrected_label"] == "spam"
    assert body["model_version"] == TEST_MODEL_VERSION
    with sqlite3.connect(db_path) as connection:
        stored = connection.execute(
            "SELECT corrected_label FROM feedback WHERE id = ?", (body["id"],)
        ).fetchone()
    assert stored == ("spam",)


def test_feedback_rejects_second_correction(client: TestClient) -> None:
    """Test of the 409 response when a prediction is corrected twice."""
    prediction_id = predict(client, [SPAM_MESSAGE])["predictions"][0]["prediction_id"]
    payload = {"prediction_id": prediction_id, "corrected_label": "spam"}
    assert client.post("/feedback", json=payload).status_code == 201
    assert client.post("/feedback", json=payload).status_code == 409


def test_feedback_rejects_unknown_prediction(client: TestClient) -> None:
    """Test of the 404 response for a prediction that does not exist."""
    response = client.post("/feedback", json={"prediction_id": 999_999, "corrected_label": "spam"})
    assert response.status_code == 404


@pytest.mark.parametrize(
    "payload",
    [
        {"prediction_id": 1, "corrected_label": "ham"},
        {"prediction_id": 0, "corrected_label": "spam"},
        {"corrected_label": "spam"},
    ],
    ids=["invalid-label", "invalid-id", "missing-id"],
)
def test_feedback_rejects_invalid_input(client: TestClient, payload: dict) -> None:
    """Test of the 422 response for feedback that does not match the schema."""
    assert client.post("/feedback", json=payload).status_code == 422


def test_value_error_returns_400(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test of the 400 response for a value error raised while processing a request."""

    def fail(*args: object) -> None:
        raise ValueError("invalid value")

    monkeypatch.setattr(api, "predict_messages", fail)
    response = client.post("/predict", json={"texts": ["hello"]})
    assert response.status_code == 400
    assert response.json() == {"detail": "invalid value"}


def test_unexpected_error_returns_500_without_details(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test of the generic 500 response, which hides the internal error from the client."""

    def fail(*args: object) -> None:
        raise RuntimeError("secret internal detail")

    monkeypatch.setattr(api, "predict_messages", fail)
    response = client.post("/predict", json={"texts": ["hello"]})
    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error"}


def test_endpoints_return_503_without_model(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test of the 503 response when the model and the database are not available."""
    monkeypatch.setattr(api.app.state, "model", None)
    monkeypatch.setattr(api.app.state, "prediction_store", None)
    assert client.post("/predict", json={"texts": ["hello"]}).status_code == 503
    assert client.get("/model").status_code == 503
    feedback = {"prediction_id": 1, "corrected_label": "spam"}
    assert client.post("/feedback", json=feedback).status_code == 503
    assert client.get("/health").json()["model_loaded"] is False


def test_startup_fails_when_model_cannot_be_loaded(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test of the startup failure when the production model cannot be loaded."""

    def fail(name: str, alias: str) -> LoadedModel:
        raise ModelLoadError("registry unreachable")

    monkeypatch.setattr(api, "load_production_model", fail)
    with pytest.raises(ModelLoadError, match="registry unreachable"):
        with TestClient(api.app):
            pass
