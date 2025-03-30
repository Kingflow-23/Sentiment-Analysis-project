import os
import sqlite3

from config import DB_PATH


def init_db():
    """Initialize the SQLite database and create the logs table if it doesn't exist."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS sentiment_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            prediction INTEGER NOT NULL,
            confidence FLOAT NOT NULL,
            inference_method TEXT NOT NULL,
            num_classes INTEGER NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """
    )
    conn.commit()
    conn.close()


def log_prediction(
    text: str,
    prediction: int,
    confidence: float,
    inference_method: str,
    num_classes: int,
):
    """Insert a log entry for a sentiment prediction."""

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO sentiment_logs (text, prediction, confidence, inference_method, num_classes)
        VALUES (?, ?, ?, ?, ?)
    """,
        (text, prediction, confidence, inference_method, num_classes),
    )
    conn.commit()
    conn.close()


# Initialize the database and create the table if it doesn't exist
init_db()
