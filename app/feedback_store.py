from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
FEEDBACK_PATH = ROOT / "feedback.csv"
FEEDBACK_COLUMNS = ["userId", "movieId", "feedback", "model", "timestamp"]


def _empty_feedback() -> pd.DataFrame:
    return pd.DataFrame(columns=FEEDBACK_COLUMNS)


def _google_sheets_configured() -> bool:
    try:
        return bool(st.secrets.get("GOOGLE_SHEET_ID")) and bool(st.secrets.get("gcp_service_account"))
    except (FileNotFoundError, KeyError):
        return False


@st.cache_resource(show_spinner=False)
def _worksheet():
    import gspread
    from google.oauth2.service_account import Credentials

    service_account = dict(st.secrets["gcp_service_account"])
    service_account["private_key"] = service_account["private_key"].replace("\\n", "\n")
    credentials = Credentials.from_service_account_info(
        service_account,
        scopes=["https://www.googleapis.com/auth/spreadsheets"],
    )
    client = gspread.authorize(credentials)
    return client.open_by_key(st.secrets["GOOGLE_SHEET_ID"]).worksheet("feedback")


def _normalise(frame: pd.DataFrame) -> pd.DataFrame:
    if "feedback" not in frame.columns and "reaction" in frame.columns:
        frame["feedback"] = frame["reaction"].map({"LIKE": "up", "DISLIKE": "down"}).fillna("")
    for column in FEEDBACK_COLUMNS:
        if column not in frame.columns:
            frame[column] = ""
    frame["userId"] = pd.to_numeric(frame["userId"], errors="coerce")
    frame["movieId"] = pd.to_numeric(frame["movieId"], errors="coerce")
    frame["feedback"] = frame["feedback"].astype(str).str.lower()
    return frame.dropna(subset=["userId", "movieId"])[FEEDBACK_COLUMNS]


def _load_local() -> pd.DataFrame:
    if not FEEDBACK_PATH.is_file():
        return _empty_feedback()
    try:
        return _normalise(pd.read_csv(FEEDBACK_PATH))
    except (OSError, pd.errors.ParserError, UnicodeDecodeError, ValueError) as exc:
        st.error(f"Could not read local feedback.csv: {exc}")
        return _empty_feedback()


def load_feedback() -> pd.DataFrame:
    if not _google_sheets_configured():
        return _load_local()
    try:
        worksheet = _worksheet()
        records = worksheet.get_all_records()
        return _normalise(pd.DataFrame(records)) if records else _empty_feedback()
    except (OSError, KeyError, ValueError) as exc:
        st.error(f"Google Sheets feedback could not be read; using local fallback: {exc}")
        return _load_local()


def _record(user_id: int, movie_id: int, rating: str, model_name: str) -> list:
    return [
        user_id,
        movie_id,
        rating,
        model_name,
        datetime.now(timezone.utc).isoformat(timespec="seconds"),
    ]


def save_feedback(user_id: int, movie_id: int, rating: str, model_name: str) -> None:
    record = _record(user_id, movie_id, rating, model_name)
    if not _google_sheets_configured():
        existing = _load_local()
        keep = ~(
            (existing["userId"] == user_id)
            & (existing["movieId"] == movie_id)
            & (existing["model"].astype(str) == model_name)
        )
        pd.concat(
            [existing.loc[keep], pd.DataFrame([record], columns=FEEDBACK_COLUMNS)],
            ignore_index=True,
        ).to_csv(FEEDBACK_PATH, index=False)
        return

    try:
        worksheet = _worksheet()
        values = worksheet.get_all_values()
        if not values:
            worksheet.append_row(FEEDBACK_COLUMNS)
            values = [FEEDBACK_COLUMNS]
        if values[0] != FEEDBACK_COLUMNS:
            raise ValueError(f"Expected headers {FEEDBACK_COLUMNS}, found {values[0]}")

        target_row = None
        for row_number, row in enumerate(values[1:], start=2):
            if len(row) >= 4 and row[0] == str(user_id) and row[1] == str(movie_id) and row[3] == model_name:
                target_row = row_number
                break
        if target_row:
            worksheet.update(f"A{target_row}:E{target_row}", [record])
        else:
            worksheet.append_row(record)
    except (OSError, KeyError, ValueError) as exc:
        st.error(f"Google Sheets feedback could not be saved; using local fallback: {exc}")
        existing = _load_local()
        keep = ~(
            (existing["userId"] == user_id)
            & (existing["movieId"] == movie_id)
            & (existing["model"].astype(str) == model_name)
        )
        pd.concat(
            [existing.loc[keep], pd.DataFrame([record], columns=FEEDBACK_COLUMNS)],
            ignore_index=True,
        ).to_csv(FEEDBACK_PATH, index=False)
