"""Shared SQLite connection helper for the local reference and job services."""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager


@contextmanager
def connect(path: str):
    db = sqlite3.connect(path, timeout=5)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys=ON")
    try:
        with db:
            yield db
    finally:
        db.close()
