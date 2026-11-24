"""
Request and response schemas of the API.

Pydantic validates every request against these models before the endpoint runs, and FastAPI
returns a 422 response describing the problem when the validation fails. The schemas also
generate the interactive documentation at ``/docs`` and ``/redoc``, including the examples.
"""

from datetime import datetime, timezone
from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field

from src.api.settings import DEFAULT_SPAM_THRESHOLD, MAX_BATCH_SIZE, MAX_TEXT_LENGTH
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


class SummaryResponse(BaseModel):
    """Return of the Summari when is called"""

    summary: str


class SummaryRequest(BaseModel):
    """The request of the summarization"""

    text: MessageText


class RootResponse(BaseModel):
    """Welcome message of the API."""

    message: str
    name: str
    version: str


class HealthResponse(BaseModel):
    """Status of the service and of the loaded model."""

    status: Literal["ok"]
    model_loaded: bool
    model_name: str | None
    model_version: str | None


class ModelInfoResponse(BaseModel):
    """Metadata of the production model."""

    name: str = Field(description="Name of the model in the MLflow Model Registry.")
    alias: str = Field(description="Alias through which the model was loaded.")
    version: str = Field(description="Registered model version.")
    run_id: str | None = Field(description="MLflow run that produced the model.")
    metrics: dict[str, float] = Field(description="Cross-validation and test metrics.")


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

    model_name: str
    model_version: str
    threshold: float
    predictions: list[PredictionResult]


class FeedbackRequest(BaseModel):
    """User correction of a prediction."""

    prediction_id: int = Field(ge=1, description="Identifier returned by /predict.")
    corrected_label: Label = Field(description="Label that the user considers correct.")

    model_config = ConfigDict(
        json_schema_extra={"examples": [{"prediction_id": 1, "corrected_label": "not_spam"}]}
    )


class FeedbackResponse(BaseModel):
    """Confirmation of a stored feedback entry."""

    id: int
    prediction_id: int
    created_at: datetime
    predicted_label: Label
    corrected_label: Label
    model_version: str = Field(description="Version of the model that made the prediction.")
    message: str


def as_utc(value: datetime) -> datetime:
    """Mark as UTC a stored time read without time zone (they are always stored in UTC)."""
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


UtcDatetime = Annotated[datetime, AfterValidator(as_utc)]


class PredictionRecordRequest(BaseModel):
    """Columns of a stored prediction, to create it (POST) or replace it (PUT)."""

    text: MessageText = Field(description="Message. It is anonymized before being stored.")
    predicted_label: Label
    spam_probability: float = Field(ge=0.0, le=1.0)
    threshold: float = Field(default=DEFAULT_SPAM_THRESHOLD, ge=0.0, le=1.0)
    model_name: str = Field(min_length=1)
    model_version: str = Field(min_length=1)

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "text": "WIN a FREE prize now!!!",
                    "predicted_label": "spam",
                    "spam_probability": 0.97,
                    "threshold": 0.5,
                    "model_name": "spam-classifier",
                    "model_version": "1",
                }
            ]
        }
    )


class PredictionRecordResponse(BaseModel):
    """Row of the predictions table."""

    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

    id: int
    created_at: UtcDatetime
    text: str
    predicted_label: Label
    spam_probability: float
    threshold: float
    model_name: str
    model_version: str


class FeedbackUpdateRequest(BaseModel):
    """New corrected label of a feedback entry."""

    corrected_label: Label

    model_config = ConfigDict(json_schema_extra={"examples": [{"corrected_label": "spam"}]})


class FeedbackRecordResponse(BaseModel):
    """Row of the feedback table."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    prediction_id: int
    created_at: UtcDatetime
    corrected_label: Label
