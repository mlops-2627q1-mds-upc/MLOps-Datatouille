"""
Storage of the predictions served by the API and of the user feedback, in a SQLite database.

Two tables are kept:

- ``predictions``: every prediction returned by ``/predict``, with the anonymized message, the
  predicted label, the spam probability, the threshold, the model version and a UTC timestamp.
  This table is the input of the drift monitoring (Milestone 6).
- ``feedback``: the label that the user considers correct for a stored prediction. Each
  prediction accepts one correction. With the prediction it refers to, this table provides the
  labelled examples for the retraining of the model.

Messages are stored anonymized, never as received. A new connection is opened for each
operation, since FastAPI runs synchronous endpoints in a thread pool and a SQLite connection
must not be shared between threads.
"""

from collections.abc import Iterator, Sequence
from contextlib import closing, contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import sqlite3

from src.modeling.predict import Label, Prediction

CREATE_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    text TEXT NOT NULL,
    predicted_label TEXT NOT NULL CHECK (predicted_label IN ('spam', 'not_spam')),
    spam_probability REAL NOT NULL,
    threshold REAL NOT NULL,
    model_name TEXT NOT NULL,
    model_version TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id INTEGER NOT NULL UNIQUE REFERENCES predictions (id),
    created_at TEXT NOT NULL,
    corrected_label TEXT NOT NULL CHECK (corrected_label IN ('spam', 'not_spam'))
);
"""

INSERT_PREDICTION_SQL = """
INSERT INTO predictions
    (created_at, text, predicted_label, spam_probability, threshold, model_name, model_version)
VALUES (?, ?, ?, ?, ?, ?, ?)
"""

SELECT_PREDICTION_SQL = """
SELECT predicted_label, model_version FROM predictions WHERE id = ?
"""

INSERT_FEEDBACK_SQL = """
INSERT INTO feedback (prediction_id, created_at, corrected_label) VALUES (?, ?, ?)
"""


class PredictionNotFoundError(LookupError):
    """Raised when feedback refers to a prediction that does not exist."""


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


class PredictionStore:
    """SQLite database of served predictions and user feedback."""

    def __init__(self, db_path: Path) -> None:
        """
        Set the database file of the store.

        Args:
            db_path: SQLite file. It is created, with its parent directories, by ``initialize``.
        """
        self.db_path = db_path

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        """
        Open a connection and commit its changes on success, or roll them back on error.

        Yields:
            Connection with foreign keys enforced.
        """
        with closing(sqlite3.connect(self.db_path)) as connection, connection:
            connection.execute("PRAGMA foreign_keys = ON")
            yield connection

    def initialize(self) -> None:
        """Create the database file and its tables if they do not exist."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._transaction() as connection:
            connection.executescript(CREATE_TABLES_SQL)

    def add_predictions(
        self,
        predictions: Sequence[Prediction],
        threshold: float,
        model_name: str,
        model_version: str,
    ) -> list[int]:
        """
        Store a batch of predictions in a single transaction.

        Args:
            predictions: Predictions returned by the model, with anonymized messages.
            threshold: Decision threshold used for the batch.
            model_name: Registered name of the model.
            model_version: Registered version of the model.

        Returns:
            Identifier of each stored prediction, in the order of ``predictions``.
        """
        created_at = datetime.now(timezone.utc).isoformat()
        ids = []
        with self._transaction() as connection:
            for prediction in predictions:
                cursor = connection.execute(
                    INSERT_PREDICTION_SQL,
                    (
                        created_at,
                        prediction.text,
                        prediction.label,
                        prediction.spam_probability,
                        threshold,
                        model_name,
                        model_version,
                    ),
                )
                if cursor.lastrowid is None:
                    raise RuntimeError("SQLite did not return the id of the stored prediction")
                ids.append(cursor.lastrowid)
        return ids

    def add_feedback(self, prediction_id: int, corrected_label: Label) -> StoredFeedback:
        """
        Store the label that the user considers correct for a prediction.

        Args:
            prediction_id: Identifier returned by ``add_predictions``.
            corrected_label: Label given by the user.

        Returns:
            The stored feedback, with the predicted label and the model version.

        Raises:
            PredictionNotFoundError: If the prediction does not exist.
            FeedbackExistsError: If the prediction already has feedback.
        """
        created_at = datetime.now(timezone.utc)
        with self._transaction() as connection:
            row = connection.execute(SELECT_PREDICTION_SQL, (prediction_id,)).fetchone()
            if row is None:
                raise PredictionNotFoundError(f"Prediction {prediction_id} does not exist")
            try:
                cursor = connection.execute(
                    INSERT_FEEDBACK_SQL, (prediction_id, created_at.isoformat(), corrected_label)
                )
            except sqlite3.IntegrityError as error:
                raise FeedbackExistsError(
                    f"Prediction {prediction_id} already has feedback"
                ) from error
        if cursor.lastrowid is None:
            raise RuntimeError("SQLite did not return the id of the stored feedback")
        predicted_label, model_version = row
        return StoredFeedback(
            id=cursor.lastrowid,
            prediction_id=prediction_id,
            created_at=created_at,
            predicted_label=predicted_label,
            corrected_label=corrected_label,
            model_version=model_version,
        )
