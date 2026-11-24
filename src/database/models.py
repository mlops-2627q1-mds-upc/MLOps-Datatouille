from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Text,
)
from sqlalchemy.orm import (
    Mapped,
    mapped_column,
    relationship,
)

from src.database.base import Base

LABEL_CHECK = "IN ('spam', 'not_spam')"


def utc_now() -> datetime:
    """Return the current time in UTC, with time zone."""
    return datetime.now(timezone.utc)


class PredictionRecord(Base):
    """Prediction returned by the API."""

    __tablename__ = "predictions"
    __table_args__ = (
        CheckConstraint(f"predicted_label {LABEL_CHECK}", name="ck_predictions_predicted_label"),
        Index("ix_predictions_created_at", "created_at"),
        Index("ix_predictions_model_version", "model_version"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )
    text: Mapped[str] = mapped_column(Text, nullable=False)
    predicted_label: Mapped[str] = mapped_column(Text, nullable=False)
    spam_probability: Mapped[float] = mapped_column(Float, nullable=False)
    threshold: Mapped[float] = mapped_column(Float, nullable=False)
    model_name: Mapped[str] = mapped_column(Text, nullable=False)
    model_version: Mapped[str] = mapped_column(Text, nullable=False)

    # Deleting a prediction also deletes its feedback
    feedback: Mapped[FeedbackRecord | None] = relationship(
        back_populates="prediction", uselist=False, cascade="all, delete-orphan"
    )


class FeedbackRecord(Base):
    """Label that the user considers correct for a stored prediction."""

    __tablename__ = "feedback"
    __table_args__ = (
        CheckConstraint(f"corrected_label {LABEL_CHECK}", name="ck_feedback_corrected_label"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    prediction_id: Mapped[int] = mapped_column(
        ForeignKey("predictions.id", ondelete="CASCADE"), nullable=False, unique=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )
    corrected_label: Mapped[str] = mapped_column(Text, nullable=False)

    prediction: Mapped[PredictionRecord] = relationship(back_populates="feedback")
