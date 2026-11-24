"""
Unit tests for the API endpoints, run through FastAPI's ``TestClient``.

The production model is replaced by a small classifier trained in ``conftest.py`` and Gemini
is mocked, so the tests need neither network access nor credentials. The prediction database
is created in a temporary directory.
"""

from collections.abc import Iterator
from pathlib import Path
import sqlite3
from unittest.mock import MagicMock

from fastapi.testclient import TestClient
import httpx
import httpx2
import pytest
from sklearn.pipeline import Pipeline

from src.api import api
from src.api.model_loader import LoadedModel, ModelLoadError
from src.api.settings import (
    API_VERSION,
    CLOUD_CALLS_TIMEOUT,
    MAX_BATCH_SIZE,
    MAX_TEXT_LENGTH,
)

SPAM_MESSAGE = "WIN a FREE prize now!!! Claim your cash reward at http://win.example"
NOT_SPAM_MESSAGE = "Hi team, the project meeting has been moved to Monday at 10."
CONTACT_MESSAGE = "Thanks for the update. Write to ana@example.com or call +34 612 345 678."
TEST_MODEL_VERSION = "7"
CLOUD_URL = "https://cloud.example/summarize"


@pytest.fixture(scope="module")
def db_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Temporary SQLite file of predictions and feedback, shared by the module."""
    return tmp_path_factory.mktemp("api") / "predictions.db"


@pytest.fixture(scope="module")
def client(spam_pipeline: Pipeline, db_path: Path) -> Iterator[TestClient]:
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
        gemini = MagicMock()
        gemini.models.generate_content.return_value.text = "Gemini summary"
        patch.setattr(api.genai, "Client", MagicMock(return_value=gemini))
        patch.setattr(api, "CLOUD_SUMMARIZATION_URL", CLOUD_URL)
        cloud_response = httpx.Response(
            200,
            json={"summary": "Cloud summary", "latency": 1.0},
            request=httpx.Request("POST", CLOUD_URL),
        )
        patch.setattr(api.httpx, "post", MagicMock(return_value=cloud_response))
        with TestClient(api.app, raise_server_exceptions=False) as test_client:
            yield test_client


@pytest.fixture
def summary_calls(client: TestClient) -> None:
    """Clear the recorded calls to the simulated Gemini and cloud services."""
    api.genai.Client.reset_mock()
    api.httpx.post.reset_mock()


def sent_to_gemini() -> str:
    """Return the content of the only request sent to the simulated Gemini."""
    generate_content = api.genai.Client.return_value.models.generate_content
    generate_content.assert_called_once()
    return generate_content.call_args.kwargs["contents"]


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


@pytest.mark.parametrize("query", ["", "?summary=false", "?cloud=true"])
def test_predict_omits_summary_unless_requested(
    client: TestClient, summary_calls: None, query: str
) -> None:
    """Without summary=true there is no summary key and no summarizer is called."""
    response = client.post(f"/predict{query}", json={"texts": [SPAM_MESSAGE]})
    assert response.status_code == 200
    [result] = response.json()["predictions"]
    assert set(result) == {"prediction_id", "label", "confidence", "spam_probability"}
    api.genai.Client.assert_not_called()
    api.httpx.post.assert_not_called()


def test_predict_uses_mocked_gemini(client: TestClient, summary_calls: None) -> None:
    """A requested summary comes from the simulated Gemini client, without using cloud."""
    response = client.post("/predict?summary=true", json={"texts": [SPAM_MESSAGE]})
    assert response.status_code == 200
    assert response.json()["predictions"][0]["summary"] == "Gemini summary"
    api.genai.Client.assert_called_once()
    assert SPAM_MESSAGE in sent_to_gemini()
    api.httpx.post.assert_not_called()


def test_predict_summarizes_batch_in_order_with_one_client(
    client: TestClient, summary_calls: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each message of a batch gets its own summary and the Gemini client is reused."""
    generate_content = api.genai.Client.return_value.models.generate_content
    monkeypatch.setattr(
        generate_content, "side_effect", [MagicMock(text="First"), MagicMock(text="Second")]
    )
    response = client.post(
        "/predict?summary=true", json={"texts": [SPAM_MESSAGE, NOT_SPAM_MESSAGE]}
    )
    assert response.status_code == 200
    summaries = [result["summary"] for result in response.json()["predictions"]]
    assert summaries == ["First", "Second"]
    contents = [call.kwargs["contents"] for call in generate_content.call_args_list]
    assert SPAM_MESSAGE in contents[0] and NOT_SPAM_MESSAGE in contents[1]
    api.genai.Client.assert_called_once()


def test_predict_rejects_invalid_summary_flag(client: TestClient) -> None:
    """Test of the 422 response when the summary flag is not a boolean."""
    response = client.post("/predict?summary=maybe", json={"texts": [SPAM_MESSAGE]})
    assert response.status_code == 422


def test_predict_cloud_directly(client: TestClient, summary_calls: None) -> None:
    """Cloud mode skips Gemini and sends the original message as JSON."""
    response = client.post("/predict?summary=true&cloud=true", json={"texts": [CONTACT_MESSAGE]})
    assert response.status_code == 200
    assert response.json()["predictions"][0]["summary"] == "Cloud summary"
    api.genai.Client.assert_not_called()
    api.httpx.post.assert_called_once_with(
        CLOUD_URL,
        json={"text": CONTACT_MESSAGE},
        timeout=CLOUD_CALLS_TIMEOUT,
    )


@pytest.mark.parametrize("code", [404, 503])
def test_predict_falls_back_on_gemini_error(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, code: int
) -> None:
    """Gemini API errors trigger the cloud fallback."""
    gemini = api.genai.Client.return_value
    monkeypatch.setattr(
        gemini.models.generate_content,
        "side_effect",
        api.errors.APIError(code, {"error": {"code": code, "message": "unavailable"}}),
    )
    response = client.post("/predict?summary=true", json={"texts": [SPAM_MESSAGE]})
    assert response.status_code == 200
    assert response.json()["predictions"][0]["summary"] == "Cloud summary"


def test_predict_falls_back_on_empty_gemini(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An empty Gemini summary is retried using cloud."""
    monkeypatch.setattr(
        api.genai.Client.return_value.models.generate_content.return_value, "text", ""
    )
    response = client.post("/predict?summary=true", json={"texts": [SPAM_MESSAGE]})
    assert response.status_code == 200
    assert response.json()["predictions"][0]["summary"] == "Cloud summary"


def test_predict_falls_back_when_gemini_client_fails(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unavailable Gemini client does not prevent cloud summarization."""
    monkeypatch.setattr(api.genai.Client, "side_effect", ValueError("missing key"))
    response = client.post("/predict?summary=true", json={"texts": [SPAM_MESSAGE]})
    assert response.status_code == 200
    assert response.json()["predictions"][0]["summary"] == "Cloud summary"


@pytest.mark.parametrize("payload", [None, {}, {"summary": ""}, {"summary": 1}])
def test_predict_handles_invalid_cloud_response(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, payload: object
) -> None:
    """An invalid cloud response leaves the prediction intact without a summary."""
    response = httpx.Response(200, json=payload, request=httpx.Request("POST", CLOUD_URL))
    monkeypatch.setattr(api.httpx.post, "return_value", response)
    result = client.post("/predict?summary=true&cloud=true", json={"texts": [SPAM_MESSAGE]})
    assert result.status_code == 200
    assert result.json()["predictions"][0]["summary"] is None
    assert result.json()["predictions"][0]["prediction_id"] >= 1


def test_predict_handles_both_services_failing(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Failures in both summarizers are logged without losing the classification."""
    monkeypatch.setattr(api.genai.Client, "side_effect", ValueError("missing key"))
    monkeypatch.setattr(api.httpx.post, "side_effect", httpx.ConnectError("unavailable"))
    response = client.post("/predict?summary=true", json={"texts": [SPAM_MESSAGE]})
    assert response.status_code == 200
    assert response.json()["predictions"][0]["summary"] is None
    assert "Gemini client unavailable" in caplog.text
    assert "Cloud summarization request failed" in caplog.text


@pytest.mark.parametrize(
    "cloud_response",
    [
        httpx.Response(503, request=httpx.Request("POST", CLOUD_URL)),
        httpx.Response(
            200,
            content=b"not json",
            request=httpx.Request("POST", CLOUD_URL),
        ),
    ],
)
def test_predict_handles_cloud_http_and_json_errors(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, cloud_response: httpx.Response
) -> None:
    """Cloud HTTP failures and malformed JSON do not prevent classification."""
    monkeypatch.setattr(api.httpx.post, "return_value", cloud_response)
    response = client.post("/predict?summary=true&cloud=true", json={"texts": [SPAM_MESSAGE]})
    assert response.status_code == 200
    assert response.json()["predictions"][0]["summary"] is None


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


def test_summarize_uses_gemini(client: TestClient, summary_calls: None) -> None:
    """Test of a summary generated by Gemini, without using cloud."""
    response = client.post("/summarize", json={"text": NOT_SPAM_MESSAGE})
    assert response.status_code == 200
    assert response.json() == {"summary": "Gemini summary"}
    assert NOT_SPAM_MESSAGE in sent_to_gemini()
    api.httpx.post.assert_not_called()


def test_summarize_cloud_directly(client: TestClient, summary_calls: None) -> None:
    """Cloud mode skips Gemini and sends the original message as JSON."""
    response = client.post("/summarize?cloud=true", json={"text": CONTACT_MESSAGE})
    assert response.status_code == 200
    assert response.json() == {"summary": "Cloud summary"}
    api.genai.Client.assert_not_called()
    api.httpx.post.assert_called_once_with(
        CLOUD_URL, json={"text": CONTACT_MESSAGE}, timeout=CLOUD_CALLS_TIMEOUT
    )


@pytest.mark.parametrize(
    ("endpoint", "payload"),
    [
        ("/summarize", {"text": CONTACT_MESSAGE}),
        ("/predict?summary=true", {"texts": [CONTACT_MESSAGE]}),
    ],
    ids=["summarize", "predict"],
)
def test_gemini_receives_anonymized_message(
    client: TestClient, summary_calls: None, endpoint: str, payload: dict
) -> None:
    """Contact details are anonymized before a message is sent to Gemini."""
    assert client.post(endpoint, json=payload).status_code == 200
    contents = sent_to_gemini()
    assert "xxemail" in contents and "xxphone" in contents
    assert "ana@example.com" not in contents and "612" not in contents


@pytest.mark.parametrize(
    "error",
    [
        api.errors.APIError(503, {"error": {"code": 503, "message": "unavailable"}}),
        httpx2.ReadTimeout("timed out"),
        httpx2.ConnectError("unreachable"),
        httpx.ReadTimeout("timed out"),
        TimeoutError("timed out"),
        ValueError("invalid"),
    ],
    ids=["api", "httpx2-timeout", "httpx2-connect", "httpx-timeout", "timeout", "value"],
)
@pytest.mark.parametrize(
    ("endpoint", "payload"),
    [
        ("/summarize", {"text": CONTACT_MESSAGE}),
        ("/predict?summary=true", {"texts": [CONTACT_MESSAGE]}),
    ],
    ids=["summarize", "predict"],
)
def test_gemini_failure_falls_back_to_cloud(
    client: TestClient,
    summary_calls: None,
    monkeypatch: pytest.MonkeyPatch,
    endpoint: str,
    payload: dict,
    error: Exception,
) -> None:
    """Gemini errors, including timeouts, fall back to cloud with the original message."""
    generate_content = api.genai.Client.return_value.models.generate_content
    monkeypatch.setattr(generate_content, "side_effect", error)
    response = client.post(endpoint, json=payload)
    assert response.status_code == 200
    assert "Cloud summary" in response.text
    api.httpx.post.assert_called_once_with(
        CLOUD_URL, json={"text": CONTACT_MESSAGE}, timeout=CLOUD_CALLS_TIMEOUT
    )


@pytest.mark.parametrize("text", [None, "", "   "])
def test_summarize_falls_back_on_empty_gemini(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, text: str | None
) -> None:
    """An empty Gemini summary is retried using cloud."""
    gemini_response = api.genai.Client.return_value.models.generate_content.return_value
    monkeypatch.setattr(gemini_response, "text", text)
    response = client.post("/summarize", json={"text": NOT_SPAM_MESSAGE})
    assert response.status_code == 200
    assert response.json() == {"summary": "Cloud summary"}


def test_summarize_falls_back_when_gemini_client_fails(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unavailable Gemini client does not prevent cloud summarization."""
    monkeypatch.setattr(api.genai.Client, "side_effect", ValueError("missing key"))
    response = client.post("/summarize", json={"text": NOT_SPAM_MESSAGE})
    assert response.status_code == 200
    assert response.json() == {"summary": "Cloud summary"}


@pytest.mark.parametrize("query", ["", "?cloud=true"])
@pytest.mark.parametrize(
    "cloud_error",
    [httpx.ConnectError("unavailable"), httpx.ReadTimeout("timed out")],
    ids=["connect", "timeout"],
)
def test_summarize_returns_502_when_all_services_fail(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, query: str, cloud_error: Exception
) -> None:
    """Test of the 502 response when no service returns a summary."""
    monkeypatch.setattr(api.genai.Client, "side_effect", ValueError("missing key"))
    monkeypatch.setattr(api.httpx.post, "side_effect", cloud_error)
    response = client.post(f"/summarize{query}", json={"text": NOT_SPAM_MESSAGE})
    assert response.status_code == 502
    assert response.json() == {"detail": "No summarization service returned a summary"}


def test_summarize_returns_502_when_cloud_is_not_configured(
    client: TestClient, summary_calls: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test of the 502 response in cloud mode without a configured URL."""
    monkeypatch.setattr(api, "CLOUD_SUMMARIZATION_URL", None)
    response = client.post("/summarize?cloud=true", json={"text": NOT_SPAM_MESSAGE})
    assert response.status_code == 502
    api.httpx.post.assert_not_called()


@pytest.mark.parametrize(
    "payload",
    [
        {"text": ""},
        {"text": "   "},
        {"text": "a" * (MAX_TEXT_LENGTH + 1)},
        {"text": None},
        {"text": ["hello"]},
        {},
    ],
    ids=["empty", "blank", "too-long", "null", "not-a-string", "missing"],
)
def test_summarize_rejects_invalid_input(
    client: TestClient, summary_calls: None, payload: dict
) -> None:
    """Test of the 422 response for requests that do not match the schema."""
    assert client.post("/summarize", json=payload).status_code == 422
    api.genai.Client.assert_not_called()
    api.httpx.post.assert_not_called()


def test_summarize_works_without_model(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Summaries do not depend on the classifier or the prediction database."""
    monkeypatch.setattr(api.app.state, "model", None)
    monkeypatch.setattr(api.app.state, "prediction_store", None)
    response = client.post("/summarize", json={"text": NOT_SPAM_MESSAGE})
    assert response.status_code == 200


def test_create_gemini_client_sets_timeout_in_milliseconds(
    client: TestClient, summary_calls: None
) -> None:
    """The shared timeout, in seconds, is converted to the milliseconds Gemini expects."""
    assert api.create_gemini_client() is api.genai.Client.return_value
    options = api.genai.Client.call_args.kwargs["http_options"]
    assert options.timeout == CLOUD_CALLS_TIMEOUT * 1000


def test_create_gemini_client_returns_none_on_error(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A client that cannot be created is logged and reported as None."""
    monkeypatch.setattr(api.genai.Client, "side_effect", ValueError("missing key"))
    assert api.create_gemini_client() is None
    assert "Gemini client unavailable" in caplog.text


def test_generate_summary_without_client_uses_cloud(
    client: TestClient, summary_calls: None
) -> None:
    """Without a Gemini client the summary comes from cloud."""
    assert api.generate_summary(None, NOT_SPAM_MESSAGE) == "Cloud summary"
    api.httpx.post.assert_called_once()


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
