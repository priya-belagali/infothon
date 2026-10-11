"""Tiny helper so every file opens the same SQLite database."""
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "supply.db"


def get_conn():
    return sqlite3.connect(DB_PATH)
