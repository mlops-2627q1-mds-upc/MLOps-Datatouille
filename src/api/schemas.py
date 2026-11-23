"""
Request and response schemas of the API.

Pydantic validates every request against these models before the endpoint runs, and FastAPI
returns a 422 response describing the problem when the validation fails. The schemas also
generate the interactive documentation at ``/docs`` and ``/redoc``, including the examples.
"""

from datetime import datetime
from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field

from src.api.settings import (
    API_TITLE,
    API_VERSION,
    DEFAULT_SPAM_THRESHOLD,
    MAX_BATCH_SIZE,
    MAX_TEXT_LENGTH,
)
from src.modeling.predict import Label


def check_not_blank(text: str) -> str:
    """
    Reject messages made only of whitespace.

    Args:
        text: Message to validate.

    Returns:
        The message, unchanged.

    Raises:
        ValueError: If the message has no visible character.
    """
    if not text.strip():
        raise ValueError("The message must contain at least one non-whitespace character")
    return text


MessageText = Annotated[
    str,
    Field(min_length=1, max_length=MAX_TEXT_LENGTH, description="Raw message to classify."),
    AfterValidator(check_not_blank),
]


class ErrorResponse(BaseModel):
    """Error returned by the API when a request cannot be served."""

    detail: str = Field(description="Human-readable description of the error.")

    model_config = ConfigDict(json_schema_extra={"examples": [{"detail": "Model not loaded"}]})


class RootResponse(BaseModel):
    """Welcome message of the API."""

    message: str = Field(description="Welcome message with a pointer to the documentation.")
    name: str = Field(description="Name of the API.")
    version: str = Field(description="Version of the API.")

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "message": f"Welcome to the {API_TITLE}. See /docs for the documentation.",
                    "name": API_TITLE,
                    "version": API_VERSION,
                }
            ]
        }
    )


class HealthResponse(BaseModel):
    """Status of the service and of the loaded model."""

    status: Literal["ok"] = Field(description="Always `ok` when the service answers.")
    model_loaded: bool = Field(description="Whether predictions can be served.")
    model_name: str | None = Field(description="Name of the loaded model, if any.")
    model_version: str | None = Field(description="Version of the loaded model, if any.")

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "status": "ok",
                    "model_loaded": True,
                    "model_name": "spam-classifier",
                    "model_version": "1",
                }
            ]
        }
    )


class ModelInfoResponse(BaseModel):
    """Metadata of the production model."""

    name: str = Field(description="Name of the model in the MLflow Model Registry.")
    alias: str = Field(description="Alias through which the model was loaded.")
    version: str = Field(description="Registered model version.")
    run_id: str | None = Field(description="MLflow run that produced the model.")
    metrics: dict[str, float] = Field(description="Cross-validation and test metrics.")

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "name": "spam-classifier",
                    "alias": "production",
                    "version": "1",
                    "run_id": "3f2b9c1d8e7a4b6c9d0e1f2a3b4c5d6e",
                    "metrics": {"cv_f1_macro": 0.97, "test_f1_macro": 0.96},
                }
            ]
        }
    )


class PredictionRequest(BaseModel):
    """One or several messages to classify."""

    texts: list[MessageText] = Field(
        min_length=1,
        max_length=MAX_BATCH_SIZE,
        description=f"Messages to classify, between 1 and {MAX_BATCH_SIZE} per request.",
    )
    threshold: float = Field(
        default=DEFAULT_SPAM_THRESHOLD,
        ge=0.0,
        le=1.0,
        description=(
            "Spam probability from which a message is labelled as spam. A higher value "
            "produces fewer false positives and more false negatives."
        ),
    )

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "texts": [
                        "Congratulations! You won a FREE iPhone. Claim it at http://win.example",
                        "Hi team, the meeting has been moved to 3pm. Write to ana@example.com",
                    ],
                    "threshold": 0.5,
                }
            ]
        }
    )


class PredictionResult(BaseModel):
    """Prediction for one message."""

    prediction_id: int = Field(description="Identifier to send with /feedback.")
    label: Label = Field(description="Predicted label.")
    confidence: float = Field(ge=0.0, le=1.0, description="Probability of the predicted label.")
    spam_probability: float = Field(ge=0.0, le=1.0, description="Probability of spam.")
    summary: str | None = Field(
        default=None,
        description="Generated summary of the original message.",
    )


class PredictionResponse(BaseModel):
    """Predictions for a batch of messages, in the order of the request."""

    model_name: str = Field(description="Name of the model that made the predictions.")
    model_version: str = Field(description="Version of the model that made the predictions.")
    threshold: float = Field(description="Spam threshold applied to the batch.")
    predictions: list[PredictionResult] = Field(
        description="One result per message, in the order of the request."
    )

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "model_name": "spam-classifier",
                    "model_version": "1",
                    "threshold": 0.5,
                    "predictions": [
                        {
                            "prediction_id": 1,
                            "label": "spam",
                            "confidence": 0.98,
                            "spam_probability": 0.98,
                            "summary": "A message claims the recipient won a free phone.",
                        },
                        {
                            "prediction_id": 2,
                            "label": "not_spam",
                            "confidence": 0.97,
                            "spam_probability": 0.03,
                            "summary": "The team meeting has been moved to 3pm.",
                        },
                    ],
                }
            ]
        }
    )


class FeedbackRequest(BaseModel):
    """User correction of a prediction."""

    prediction_id: int = Field(ge=1, description="Identifier returned by /predict.")
    corrected_label: Label = Field(description="Label that the user considers correct.")

    model_config = ConfigDict(
        json_schema_extra={"examples": [{"prediction_id": 1, "corrected_label": "not_spam"}]}
    )


class FeedbackResponse(BaseModel):
    """Confirmation of a stored feedback entry."""

    id: int = Field(description="Identifier of the stored feedback entry.")
    prediction_id: int = Field(description="Prediction that the feedback corrects.")
    created_at: datetime = Field(description="UTC timestamp of the feedback.")
    predicted_label: Label = Field(description="Label returned by the model.")
    corrected_label: Label = Field(description="Label that the user considers correct.")
    model_version: str = Field(description="Version of the model that made the prediction.")
    message: str = Field(description="Confirmation message.")

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "id": 1,
                    "prediction_id": 1,
                    "created_at": "2026-11-10T10:00:00+00:00",
                    "predicted_label": "spam",
                    "corrected_label": "not_spam",
                    "model_version": "1",
                    "message": "Feedback stored",
                }
            ]
        }
    )
