"""
Storage of the predictions served by the API and of the user feedback, in PostgreSQL.

Two tables are kept (see ``src.database.models``):

- ``predictions``: every prediction returned by ``/predict``, with the anonymized message. Input
  of the drift monitoring.
- ``feedback``: the label that the user considers correct for a stored prediction. Each
  prediction accepts one correction. Together with the predictions, the labelled examples for
  the retraining of the model.

Each operation opens its own session, since FastAPI runs synchronous endpoints in a thread pool.
"""

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import cast

from sqlalchemy import select
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from src.database.base import Base
from src.database.connection import create_database_engine, create_session_factory
from src.database.models import FeedbackRecord, PredictionRecord
from src.modeling.predict import Label, Prediction


class PredictionNotFoundError(LookupError):
    """Raised when a prediction does not exist."""


class FeedbackNotFoundError(LookupError):
    """Raised when a feedback entry does not exist."""


class FeedbackExistsError(RuntimeError):
    """Raised when a prediction already has feedback."""


@dataclass(frozen=True)
class StoredFeedback:
    """
    Feedback entry after it has been stored.

    Attributes:
        id: Primary key of the feedback entry.
        prediction_id: Prediction that the feedback corrects.
        created_at: UTC time at which the feedback was stored.
        predicted_label: Label returned by the model for that prediction.
        corrected_label: Label given by the user.
        model_version: Registered version of the model that made the prediction.
    """

    id: int
    prediction_id: int
    created_at: datetime
    predicted_label: Label
    corrected_label: Label
    model_version: str


class PostgresPredictionStore:
    """PostgreSQL database of served predictions and user feedback."""

    def __init__(self, engine: Engine | None = None) -> None:
        """Set the engine of the store."""
        self.engine = engine or create_database_engine()
        self._session_factory = create_session_factory(self.engine)

    @contextmanager
    def _transaction(self) -> Iterator[Session]:
        """Open a session and commit its changes on success, or roll them back on error."""
        with self._session_factory() as session, session.begin():
            yield session

    def initialize(self) -> None:
        """Create the tables if they do not exist."""
        Base.metadata.create_all(self.engine)

    def add_predictions(
        self,
        predictions: Sequence[Prediction],
        threshold: float,
        model_name: str,
        model_version: str,
    ) -> list[int]:
        """Store a batch of predictions in a single transaction."""
        created_at = datetime.now(timezone.utc)
        records = [
            PredictionRecord(
                created_at=created_at,
                text=prediction.text,
                predicted_label=prediction.label,
                spam_probability=prediction.spam_probability,
                threshold=threshold,
                model_name=model_name,
                model_version=model_version,
            )
            for prediction in predictions
        ]
        with self._transaction() as session:
            session.add_all(records)
            session.flush()
            return [record.id for record in records]

    def add_feedback(self, prediction_id: int, corrected_label: Label) -> StoredFeedback:
        """Store the label that the user considers correct for a prediction."""
        created_at = datetime.now(timezone.utc)
        try:
            with self._transaction() as session:
                row = session.execute(
                    select(PredictionRecord.predicted_label, PredictionRecord.model_version).where(
                        PredictionRecord.id == prediction_id
                    )
                ).one_or_none()
                if row is None:
                    raise PredictionNotFoundError(f"Prediction {prediction_id} does not exist")
                feedback = FeedbackRecord(
                    prediction_id=prediction_id,
                    created_at=created_at,
                    corrected_label=corrected_label,
                )
                session.add(feedback)
                session.flush()
                feedback_id = feedback.id
        except IntegrityError as error:
            raise FeedbackExistsError(
                f"Prediction {prediction_id} already has feedback"
            ) from error
        predicted_label, model_version = row
        return StoredFeedback(
            id=feedback_id,
            prediction_id=prediction_id,
            created_at=created_at,
            # The CHECK constraint of the table only admits the two labels
            predicted_label=cast(Label, predicted_label),
            corrected_label=corrected_label,
            model_version=model_version,
        )

    # CRUD of predictions

    def list_predictions(self, limit: int = 100, offset: int = 0) -> list[PredictionRecord]:
        """Return stored predictions ordered by id, one page at a time."""
        query = select(PredictionRecord).order_by(PredictionRecord.id).limit(limit).offset(offset)
        with self._transaction() as session:
            return list(session.scalars(query))

    def get_prediction(self, prediction_id: int) -> PredictionRecord:
        """Return one prediction, or raise PredictionNotFoundError."""
        with self._transaction() as session:
            return self._get_prediction(session, prediction_id)

    def create_prediction(self, **fields: object) -> PredictionRecord:
        """Store one prediction given its columns (text, labels, probabilities, model)."""
        record = PredictionRecord(created_at=datetime.now(timezone.utc), **fields)
        with self._transaction() as session:
            session.add(record)
            session.flush()
        return record

    def update_prediction(self, prediction_id: int, **fields: object) -> PredictionRecord:
        """Replace the editable columns of a prediction, or raise PredictionNotFoundError."""
        with self._transaction() as session:
            record = self._get_prediction(session, prediction_id)
            for name, value in fields.items():
                setattr(record, name, value)
            session.flush()
        return record

    def delete_prediction(self, prediction_id: int) -> None:
        """Delete a prediction and its feedback, or raise PredictionNotFoundError."""
        with self._transaction() as session:
            session.delete(self._get_prediction(session, prediction_id))

    @staticmethod
    def _get_prediction(session: Session, prediction_id: int) -> PredictionRecord:
        record = session.get(PredictionRecord, prediction_id)
        if record is None:
            raise PredictionNotFoundError(f"Prediction {prediction_id} does not exist")
        return record

    # CRUD of feedback (creation is add_feedback)

    def list_feedback(self, limit: int = 100, offset: int = 0) -> list[FeedbackRecord]:
        """Return stored feedback ordered by id, one page at a time."""
        query = select(FeedbackRecord).order_by(FeedbackRecord.id).limit(limit).offset(offset)
        with self._transaction() as session:
            return list(session.scalars(query))

    def get_feedback(self, feedback_id: int) -> FeedbackRecord:
        """Return one feedback entry, or raise FeedbackNotFoundError."""
        with self._transaction() as session:
            return self._get_feedback(session, feedback_id)

    def update_feedback(self, feedback_id: int, corrected_label: Label) -> FeedbackRecord:
        """Change the corrected label of a feedback entry, or raise FeedbackNotFoundError."""
        with self._transaction() as session:
            record = self._get_feedback(session, feedback_id)
            record.corrected_label = corrected_label
            session.flush()
        return record

    def delete_feedback(self, feedback_id: int) -> None:
        """Delete a feedback entry, or raise FeedbackNotFoundError."""
        with self._transaction() as session:
            session.delete(self._get_feedback(session, feedback_id))

    @staticmethod
    def _get_feedback(session: Session, feedback_id: int) -> FeedbackRecord:
        record = session.get(FeedbackRecord, feedback_id)
        if record is None:
            raise FeedbackNotFoundError(f"Feedback {feedback_id} does not exist")
        return record
