"""Parquet-backed conversation and feedback log.

Captures each Q&A exchange (question, answer, references, user) and lets
feedback (star ratings, comments) be attached afterwards. Replaces the
ad-hoc logging in the original Streamlit app with a small class that is
safe to call from any host process.

Requires pandas and a parquet engine (pyarrow) — both in requirements.txt.
"""

import hashlib
import os
from datetime import datetime

import pandas as pd

COLUMNS = [
    "chat_id",
    "username",
    "question",
    "response",
    "references",
    "stars",
    "comment",
    "timestamp",
]


def make_chat_id(username: str, question: str) -> str:
    """Deterministic-ish unique id for one chat exchange."""
    seed = f"{username}-{question.strip()[:40]}-{datetime.now().isoformat(timespec='seconds')}"
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()[:12].upper()


class FeedbackLog:
    def __init__(self, path: str = "logs/userlog.parquet"):
        self.path = path

    def _read(self) -> pd.DataFrame:
        try:
            return pd.read_parquet(self.path)
        except FileNotFoundError:
            return pd.DataFrame(columns=COLUMNS)

    def _write(self, df: pd.DataFrame) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        df.to_parquet(self.path, index=False)

    def record(
        self,
        question: str,
        response: str,
        username: str = "guest",
        references: list | None = None,
    ) -> str:
        """Append a new chat exchange; returns its chat_id for later feedback."""
        df = self._read()
        chat_id = make_chat_id(username, question)
        row = pd.DataFrame(
            [
                {
                    "chat_id": chat_id,
                    "username": username,
                    "question": question,
                    "response": response,
                    "references": (
                        "; ".join(str(r) for r in references) if references else ""
                    ),
                    "stars": "",
                    "comment": "",
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                }
            ]
        )
        self._write(pd.concat([df, row], ignore_index=True))
        return chat_id

    def update_feedback(self, chat_id: str, stars: int | None = None, comment: str | None = None) -> bool:
        """Attach feedback to a previously recorded exchange. Returns True if found."""
        df = self._read()
        mask = df["chat_id"] == chat_id
        if not mask.any():
            return False
        if stars is not None:
            df.loc[mask, "stars"] = str(stars)
        if comment is not None:
            df.loc[mask, "comment"] = comment
        self._write(df)
        return True

    def to_frame(self) -> pd.DataFrame:
        """Full log as a DataFrame, for analysis of ratings/usage."""
        return self._read()
