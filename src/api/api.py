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

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse

from src.api.model_loader import LoadedModel, ModelLoadError, load_production_model
from src.api.prediction_store import (
    FeedbackExistsError,
    PredictionNotFoundError,
    PredictionStore,
)
from src.api.schemas import (
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
    PredictionResult,
    RootResponse,
)
from src.api.settings import (
    API_TITLE,
    API_VERSION,
    MLFLOW_MODEL_ALIAS,
    MLFLOW_MODEL_NAME,
    PREDICTIONS_DB_PATH,
)
from src.modeling.predict import predict_messages

logger = logging.getLogger(__name__)

API_DESCRIPTION = """
Spam detection service of the Intelligent Message Analysis Platform (IMAP).

* **Prediction**: classify one or several messages as `spam` or `not_spam`, with the
  probability of each label. Email addresses and phone numbers are anonymized before inference.
* **Feedback**: report the correct label of a prediction, to monitor and retrain the model.
* **Model**: consult the version and the evaluation metrics of the model in production.
"""

TAGS_METADATA = [
    {"name": "General", "description": "Service information and health check."},
    {"name": "Model", "description": "Metadata of the model in production."},
    {"name": "Prediction", "description": "Spam classification of messages."},
    {"name": "Feedback", "description": "Collection of user corrections."},
]


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

    app.state.prediction_store = PredictionStore(PREDICTIONS_DB_PATH)
    app.state.prediction_store.initialize()
    logger.info("Prediction database ready at %s", PREDICTIONS_DB_PATH)

    yield

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


def get_prediction_store(request: Request) -> PredictionStore:
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


@app.post("/predict", tags=["Prediction"], response_model=PredictionResponse)
def predict(
    payload: PredictionRequest,
    model: LoadedModel = Depends(get_model),
    store: PredictionStore = Depends(get_prediction_store),
) -> PredictionResponse:
    """
    Classify one or several messages as `spam` or `not_spam`.

    Email addresses and phone numbers are anonymized before inference. Each prediction gives
    the predicted label, the probability of that label (`confidence`) and the probability of
    spam. A message is labelled as spam when its spam probability is greater than or equal to
    `threshold`. Predictions are returned in the order of `texts`.

    Every prediction is stored with the anonymized message for monitoring. Its `prediction_id`
    is the reference to send to `/feedback`.
    """
    predictions = predict_messages(model.pipeline, payload.texts, payload.threshold)
    prediction_ids = store.add_predictions(
        predictions, payload.threshold, model.name, model.version
    )
    return PredictionResponse(
        model_name=model.name,
        model_version=model.version,
        threshold=payload.threshold,
        predictions=[
            PredictionResult(
                prediction_id=prediction_id,
                label=prediction.label,
                confidence=prediction.confidence,
                spam_probability=prediction.spam_probability,
            )
            for prediction_id, prediction in zip(prediction_ids, predictions, strict=True)
        ],
    )


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
    payload: FeedbackRequest, store: PredictionStore = Depends(get_prediction_store)
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
