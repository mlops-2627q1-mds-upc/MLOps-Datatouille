"""
FastAPI application that serves the spam classifier and collects user feedback.

The production model is loaded once at startup, inside the ``lifespan`` context manager, and
kept in ``app.state`` for every request. If it cannot be loaded, the application does not
start. The endpoints only validate the request, call the inference or storage modules and
build the response.

Run locally (from the root of the repo):
    uvicorn src.api.api:app --reload
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import logging
import os
import secrets

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response, Security, status
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from google import genai
from google.genai import errors, types
import httpx
import httpx2

from src.api.model_loader import LoadedModel, ModelLoadError, load_production_model
from src.api.schemas import (
    FeedbackRecordResponse,
    FeedbackRequest,
    FeedbackResponse,
    FeedbackUpdateRequest,
    HealthResponse,
    ModelInfoResponse,
    PredictionRecordRequest,
    PredictionRecordResponse,
    PredictionRequest,
    PredictionResponse,
    PredictionResult,
    RootResponse,
    SummaryRequest,
    SummaryResponse,
)
from src.api.settings import (
    ADMIN_API_KEY,
    API_TITLE,
    API_VERSION,
    CLOUD_CALLS_TIMEOUT,
    CLOUD_SUMMARIZATION_URL,
    GEMINI_MODEL,
    MLFLOW_MODEL_ALIAS,
    MLFLOW_MODEL_NAME,
)
from src.data.preprocess import anonymize_text
from src.database.prediction_store import (
    FeedbackExistsError,
    FeedbackNotFoundError,
    PostgresPredictionStore,
    PredictionNotFoundError,
)
from src.modeling.predict import predict_messages

logger = logging.getLogger(__name__)

API_DESCRIPTION = """
Spam detection and text summarization service of the Intelligent Message Analysis Platform
(IMAP).

* **Prediction**: classify 1 to 100 messages as `spam` or `not_spam`, with the probability of
  each label. Email addresses and phone numbers are anonymized before inference.
* **Summary**: summarize messages with Google Gemini, with a cloud service as fallback, through
  `/summarize` or `/predict?summary=true`.
* **Feedback**: report the correct label of a prediction, to monitor and retrain the model.
* **Model**: consult the version and the evaluation metrics of the model in production.
* **Administration**: read, create, update and delete the stored predictions and feedback.

### Storage

Every prediction is stored in a PostgreSQL database with the anonymized message, the model
version and a UTC timestamp. Each prediction accepts one feedback entry. The data supports drift
monitoring and model retraining, and is kept across restarts and redeployments of the API.

### Authentication

The administration endpoints (tags *Admin: predictions* and *Admin: feedback*) require the
header `X-API-Key` with the key configured on the server in `ADMIN_API_KEY`. Use the
**Authorize** button to send it with every request. A missing or wrong key returns `401`, and the
endpoints return `503` when no key is configured. The other endpoints are public.

### Privacy

Messages are anonymized for classification and storage. When a summary is requested, Gemini
receives the anonymized message, while the cloud fallback service receives the **original**
message. Do not submit sensitive or confidential information. There is no rate limiting.

### Limits

Each message has 1 to 10,000 characters, with at least one non-whitespace character. The
optional `threshold` is between 0.0 and 1.0 (default 0.5).
"""

SUMMARY_CONFIG = types.GenerateContentConfig(
    system_instruction=(
        "You are an expert assistant specialized in text summarization. "
        "Summarize the text in one or two short sentences of plain text. "
        "CRITICAL: Do not use Markdown or any formatting: no asterisks, bullet points, "
        "headings or line breaks. "
        "CRITICAL: Always output the summary in the SAME language as the input text "
        "provided by the user."
    )
)

TAGS_METADATA = [
    {"name": "General", "description": "Service information and health check."},
    {"name": "Model", "description": "Metadata of the model in production."},
    {"name": "Prediction", "description": "Spam classification of messages."},
    {"name": "Feedback", "description": "Collection of user corrections."},
    {"name": "Summary", "description": "Text summarization service."},
    {
        "name": "Admin: predictions",
        "description": "CRUD of the stored predictions. Requires the `X-API-Key` header.",
    },
    {
        "name": "Admin: feedback",
        "description": "CRUD of the stored feedback. Requires the `X-API-Key` header.",
    },
]


def create_prediction_store() -> PostgresPredictionStore:
    """Create the store of predictions and feedback in the PostgreSQL of ``DATABASE_URL``."""
    return PostgresPredictionStore()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Load the production model and prepare the prediction database before serving requests.

    Args:
        app: The FastAPI application.

    Yields:
        Control to the application while it serves requests.

    Raises:
        ModelLoadError: If the production model cannot be loaded, which stops the startup.
    """
    try:
        app.state.model = load_production_model(MLFLOW_MODEL_NAME, MLFLOW_MODEL_ALIAS)
    except ModelLoadError:
        logger.exception("The API cannot start without the production model")
        raise
    logger.info(
        "Loaded model %s version %s (alias '%s')",
        app.state.model.name,
        app.state.model.version,
        app.state.model.alias,
    )

    app.state.prediction_store = create_prediction_store()
    app.state.prediction_store.initialize()
    logger.info("Prediction database (PostgreSQL) ready")

    yield

    app.state.prediction_store.engine.dispose()
    app.state.prediction_store = None
    app.state.model = None


app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    openapi_tags=TAGS_METADATA,
    lifespan=lifespan,
)


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, error: ValueError) -> JSONResponse:
    """
    Return a 400 response for invalid values detected while processing a request.

    Args:
        request: The request that failed.
        error: The raised error.

    Returns:
        JSON response with status 400 and the error message.
    """
    logger.warning("Invalid value in %s %s: %s", request.method, request.url.path, error)
    return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"detail": str(error)})


@app.exception_handler(Exception)
async def unexpected_error_handler(request: Request, error: Exception) -> JSONResponse:
    """
    Return a 500 response for unexpected errors, without exposing their details.

    The full traceback is written to the server log.

    Args:
        request: The request that failed.
        error: The raised error.

    Returns:
        JSON response with status 500 and a generic message.
    """
    logger.exception("Unexpected error in %s %s", request.method, request.url.path, exc_info=error)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"},
    )


def get_model(request: Request) -> LoadedModel:
    """
    Return the loaded production model.

    Args:
        request: The current request, which gives access to the application state.

    Returns:
        The loaded model.

    Raises:
        HTTPException: 503 if no model is loaded.
    """
    model = getattr(request.app.state, "model", None)
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model not loaded"
        )
    return model


def get_prediction_store(request: Request) -> PostgresPredictionStore:
    """
    Return the prediction store prepared at startup.

    Args:
        request: The current request, which gives access to the application state.

    Returns:
        The prediction store.

    Raises:
        HTTPException: 503 if the store is not ready.
    """
    store = getattr(request.app.state, "prediction_store", None)
    if store is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Prediction storage not ready"
        )
    return store


@app.get("/", tags=["General"], response_model=RootResponse)
def root() -> RootResponse:
    """Return a welcome message with the name and the version of the API."""
    return RootResponse(
        message=f"Welcome to the {API_TITLE}. See /docs for the documentation.",
        name=API_TITLE,
        version=API_VERSION,
    )


@app.get("/health", tags=["General"], response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    """
    Report whether the service is running and which model it serves.
    The service is running whenever this endpoint answers. `model_loaded` tells whether
    predictions can be served.
    """
    model: LoadedModel | None = getattr(request.app.state, "model", None)
    return HealthResponse(
        status="ok",
        model_loaded=model is not None,
        model_name=model.name if model else None,
        model_version=model.version if model else None,
    )


@app.get("/model", tags=["Model"], response_model=ModelInfoResponse)
def model_info(model: LoadedModel = Depends(get_model)) -> ModelInfoResponse:
    """
    Return the metadata of the model in production.
    The response includes the registered name, alias and version of the model, the MLflow run
    that produced it, and the cross-validation and test metrics recorded in that run.
    """
    return ModelInfoResponse(
        name=model.name,
        alias=model.alias,
        version=model.version,
        run_id=model.run_id,
        metrics=model.metrics,
    )


def summarize_with_cloud(text: str) -> str | None:
    """Request a summary from the configured cloud service in the Virtech."""
    if not CLOUD_SUMMARIZATION_URL:
        logger.warning("Cloud summarization URL is not configured")
        return None

    try:
        response = httpx.post(
            CLOUD_SUMMARIZATION_URL,
            json={"text": text},
            timeout=CLOUD_CALLS_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
    except httpx.HTTPStatusError as error:
        logger.warning("Cloud summarization failed with status %s", error.response.status_code)
        return None
    except httpx.RequestError as error:
        logger.warning("Cloud summarization request failed: %s", type(error).__name__)
        return None
    except ValueError:
        logger.warning("Cloud summarization returned invalid JSON")
        return None

    if not isinstance(data, dict):
        logger.warning("Cloud summarization returned an invalid response")
        return None
    summary = data.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        logger.warning("Cloud summarization returned an empty or invalid summary")
        return None
    return summary


def summarize_with_gemini(
    client: genai.Client, text: str, config: types.GenerateContentConfig
) -> str | None:
    """Request a summary from Gemini."""
    prompt = "Summarize the following text in one or two plain-text sentences:"
    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=f"{prompt}\n\n{text}",
            config=config,
        )
        summary = response.text
    except errors.APIError as error:
        logger.warning("Gemini summarization failed with status %s", error.code)
        return None
    except (httpx.RequestError, httpx2.RequestError, TimeoutError, ValueError) as error:
        logger.warning("Gemini summarization failed: %s", type(error).__name__)
        return None
    if not isinstance(summary, str) or not summary.strip():
        logger.warning("Gemini returned an empty or invalid summary")
        return None
    return summary


def create_gemini_client() -> genai.Client | None:
    """Create the Gemini client."""
    try:
        return genai.Client(
            api_key=os.getenv("GEMINI_API_KEY"),
            http_options=types.HttpOptions(timeout=int(CLOUD_CALLS_TIMEOUT * 1000)),
        )
    except (ValueError, errors.APIError) as error:
        logger.warning("Gemini client unavailable: %s", type(error).__name__)
        return None


def generate_summary(client: genai.Client | None, text: str) -> str | None:
    """Summarize a text with Gemini, falling back to the cloud service."""
    anonymized_text_to_gemini = anonymize_text(text)
    summary_text = None
    if client is not None:
        summary_text = summarize_with_gemini(client, anonymized_text_to_gemini, SUMMARY_CONFIG)
    if summary_text is None:
        summary_text = summarize_with_cloud(text)
    return summary_text


@app.post(
    "/predict",
    tags=["Prediction"],
    response_model=PredictionResponse,
    response_model_exclude_unset=True,
)
def predict(
    payload: PredictionRequest,
    cloud: bool = False,
    summary: bool = False,
    model: LoadedModel = Depends(get_model),
    store: PostgresPredictionStore = Depends(get_prediction_store),
) -> PredictionResponse:
    """
    Classify one or several messages as `spam` or `not_spam` and generate their summaries.

    Args:
        payload: Messages and classification threshold.
        cloud: Use the cloud summarizer directly instead of Gemini.
        model: Loaded classifier.
        store: Prediction storage.

    Returns:
        Predictions and their summaries.
    """
    # Classification SPAM / NOT SPAM.
    predictions = predict_messages(model.pipeline, payload.texts, payload.threshold)
    prediction_ids = store.add_predictions(
        predictions, payload.threshold, model.name, model.version
    )

    client = create_gemini_client() if summary and not cloud else None

    # Build the list of results by mapping predictions and their summaries
    results = []
    for prediction_id, prediction, original_text in zip(
        prediction_ids, predictions, payload.texts, strict=True
    ):
        # Append the prediction result along with its summary to the results list
        result_parcial = PredictionResult(
            prediction_id=prediction_id,
            label=prediction.label,
            confidence=prediction.confidence,
            spam_probability=prediction.spam_probability,
        )

        if summary:
            result_parcial.summary = generate_summary(client, original_text)
        results.append(result_parcial)

    return PredictionResponse(
        model_name=model.name,
        model_version=model.version,
        threshold=payload.threshold,
        predictions=results,
    )


@app.post(
    "/summarize",
    tags=["Summary"],
    response_model=SummaryResponse,
    responses={
        422: {"description": "Empty Blank text or too long text"},
        502: {"description": "Summary service Unavaiable"},
    },
)
def summary(payload: SummaryRequest, cloud: bool = False) -> SummaryResponse:
    """
    Summarize a message with Gemini, falling back to our cloud service.

    Args:
        payload: Message to summarize.
        cloud: Use our cloud service directly instead of Gemini.

    Returns:
        The generated summary.

    Raises:
        HTTPException: If no service returns a summary (502).
    """

    client = None if cloud else create_gemini_client()
    summary_text = generate_summary(client, payload.text)

    if summary_text is None:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="No summarization service returned a summary",
        )

    return SummaryResponse(summary=summary_text)


@app.post(
    "/feedback",
    tags=["Feedback"],
    response_model=FeedbackResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        404: {"description": "The prediction does not exist"},
        409: {"description": "The prediction already has feedback"},
    },
)
def feedback(
    payload: FeedbackRequest, store: PostgresPredictionStore = Depends(get_prediction_store)
) -> FeedbackResponse:
    """
    Store the label that the user considers correct for a prediction.

    The prediction is identified by the `prediction_id` returned by `/predict`. Each prediction
    accepts one correction. The feedback is stored with a UTC timestamp and, together with the
    stored prediction, is used to monitor and retrain the model.
    """
    try:
        stored = store.add_feedback(payload.prediction_id, payload.corrected_label)
    except PredictionNotFoundError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error
    except FeedbackExistsError as error:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error
    return FeedbackResponse(
        id=stored.id,
        prediction_id=stored.prediction_id,
        created_at=stored.created_at,
        predicted_label=stored.predicted_label,
        corrected_label=stored.corrected_label,
        model_version=stored.model_version,
        message="Feedback stored",
    )


# CRUD of the stored predictions and feedback (protected with ADMIN_API_KEY)


ADMIN_KEY_HEADER = APIKeyHeader(
    name="X-API-Key",
    scheme_name="AdminApiKey",
    description="Key of the administration endpoints, configured on the server in ADMIN_API_KEY.",
    auto_error=False,
)


def require_admin_key(x_api_key: str | None = Security(ADMIN_KEY_HEADER)) -> None:
    """
    Allow the request only if the X-API-Key header matches ``ADMIN_API_KEY``.

    Raises:
        HTTPException: 503 if no admin key is configured, 401 if the header is missing or wrong.
    """
    if not ADMIN_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Admin endpoints disabled"
        )
    if x_api_key is None or not secrets.compare_digest(x_api_key, ADMIN_API_KEY):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")


ADMIN_RESPONSES: dict[int | str, dict[str, str]] = {
    401: {"description": "Missing or invalid X-API-Key header"},
    503: {"description": "Admin endpoints disabled (ADMIN_API_KEY not set)"},
}
PAGE_LIMIT = Query(default=100, ge=1, le=500, description="Maximum number of rows to return.")
PAGE_OFFSET = Query(default=0, ge=0, description="Number of rows to skip.")


def not_found(error: LookupError) -> HTTPException:
    """Build the 404 response of a missing row."""
    return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error))


def prediction_columns(payload: PredictionRecordRequest) -> dict[str, object]:
    """Columns of a prediction from a request, with the message anonymized as in /predict."""
    columns = payload.model_dump()
    columns["text"] = anonymize_text(payload.text)
    return columns


@app.get(
    "/predictions",
    tags=["Admin: predictions"],
    response_model=list[PredictionRecordResponse],
    dependencies=[Depends(require_admin_key)],
    responses=ADMIN_RESPONSES,
)
def list_predictions(
    limit: int = PAGE_LIMIT,
    offset: int = PAGE_OFFSET,
    store: PostgresPredictionStore = Depends(get_prediction_store),
) -> list[PredictionRecordResponse]:
    """List the stored predictions, ordered by id."""
    records = store.list_predictions(limit=limit, offset=offset)
    return [PredictionRecordResponse.model_validate(record) for record in records]


@app.get(
    "/predictions/{prediction_id}",
    tags=["Admin: predictions"],
    response_model=PredictionRecordResponse,
    dependencies=[Depends(require_admin_key)],
    responses={**ADMIN_RESPONSES, 404: {"description": "The prediction does not exist"}},
)
def get_prediction(
    prediction_id: int, store: PostgresPredictionStore = Depends(get_prediction_store)
) -> PredictionRecordResponse:
    """Return one stored prediction."""
    try:
        return PredictionRecordResponse.model_validate(store.get_prediction(prediction_id))
    except PredictionNotFoundError as error:
        raise not_found(error) from error


@app.post(
    "/predictions",
    tags=["Admin: predictions"],
    response_model=PredictionRecordResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_admin_key)],
    responses=ADMIN_RESPONSES,
)
def create_prediction(
    payload: PredictionRecordRequest,
    store: PostgresPredictionStore = Depends(get_prediction_store),
) -> PredictionRecordResponse:
    """Store a prediction given all its columns. The message is anonymized before storage."""
    record = store.create_prediction(**prediction_columns(payload))
    return PredictionRecordResponse.model_validate(record)


@app.put(
    "/predictions/{prediction_id}",
    tags=["Admin: predictions"],
    response_model=PredictionRecordResponse,
    dependencies=[Depends(require_admin_key)],
    responses={**ADMIN_RESPONSES, 404: {"description": "The prediction does not exist"}},
)
def update_prediction(
    prediction_id: int,
    payload: PredictionRecordRequest,
    store: PostgresPredictionStore = Depends(get_prediction_store),
) -> PredictionRecordResponse:
    """Replace all the columns of a stored prediction (except its id and creation time)."""
    try:
        record = store.update_prediction(prediction_id, **prediction_columns(payload))
    except PredictionNotFoundError as error:
        raise not_found(error) from error
    return PredictionRecordResponse.model_validate(record)


@app.delete(
    "/predictions/{prediction_id}",
    tags=["Admin: predictions"],
    status_code=status.HTTP_204_NO_CONTENT,
    dependencies=[Depends(require_admin_key)],
    responses={**ADMIN_RESPONSES, 404: {"description": "The prediction does not exist"}},
)
def delete_prediction(
    prediction_id: int, store: PostgresPredictionStore = Depends(get_prediction_store)
) -> Response:
    """Delete a stored prediction and its feedback."""
    try:
        store.delete_prediction(prediction_id)
    except PredictionNotFoundError as error:
        raise not_found(error) from error
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@app.get(
    "/feedback",
    tags=["Admin: feedback"],
    response_model=list[FeedbackRecordResponse],
    dependencies=[Depends(require_admin_key)],
    responses=ADMIN_RESPONSES,
)
def list_feedback(
    limit: int = PAGE_LIMIT,
    offset: int = PAGE_OFFSET,
    store: PostgresPredictionStore = Depends(get_prediction_store),
) -> list[FeedbackRecordResponse]:
    """List the stored feedback, ordered by id. New feedback is created with POST /feedback."""
    records = store.list_feedback(limit=limit, offset=offset)
    return [FeedbackRecordResponse.model_validate(record) for record in records]


@app.get(
    "/feedback/{feedback_id}",
    tags=["Admin: feedback"],
    response_model=FeedbackRecordResponse,
    dependencies=[Depends(require_admin_key)],
    responses={**ADMIN_RESPONSES, 404: {"description": "The feedback does not exist"}},
)
def get_feedback(
    feedback_id: int, store: PostgresPredictionStore = Depends(get_prediction_store)
) -> FeedbackRecordResponse:
    """Return one stored feedback entry."""
    try:
        return FeedbackRecordResponse.model_validate(store.get_feedback(feedback_id))
    except FeedbackNotFoundError as error:
        raise not_found(error) from error


@app.put(
    "/feedback/{feedback_id}",
    tags=["Admin: feedback"],
    response_model=FeedbackRecordResponse,
    dependencies=[Depends(require_admin_key)],
    responses={**ADMIN_RESPONSES, 404: {"description": "The feedback does not exist"}},
)
def update_feedback(
    feedback_id: int,
    payload: FeedbackUpdateRequest,
    store: PostgresPredictionStore = Depends(get_prediction_store),
) -> FeedbackRecordResponse:
    """Change the corrected label of a stored feedback entry."""
    try:
        record = store.update_feedback(feedback_id, payload.corrected_label)
    except FeedbackNotFoundError as error:
        raise not_found(error) from error
    return FeedbackRecordResponse.model_validate(record)


@app.delete(
    "/feedback/{feedback_id}",
    tags=["Admin: feedback"],
    status_code=status.HTTP_204_NO_CONTENT,
    dependencies=[Depends(require_admin_key)],
    responses={**ADMIN_RESPONSES, 404: {"description": "The feedback does not exist"}},
)
def delete_feedback(
    feedback_id: int, store: PostgresPredictionStore = Depends(get_prediction_store)
) -> Response:
    """Delete a stored feedback entry. Its prediction is kept."""
    try:
        store.delete_feedback(feedback_id)
    except FeedbackNotFoundError as error:
        raise not_found(error) from error
    return Response(status_code=status.HTTP_204_NO_CONTENT)
