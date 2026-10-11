"""Human approval + decision log (the 'audit trail' from your slides)."""
from datetime import datetime

import pandas as pd

from core.db import get_conn


def init_table():
    conn = get_conn()
    conn.execute(
        """CREATE TABLE IF NOT EXISTS decisions (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               decided_at TEXT, order_id TEXT, action TEXT, status TEXT,
               reason TEXT, evidence TEXT, est_cost REAL)"""
    )
    conn.commit()
    conn.close()


def log_decision(order_id, action, status, reason, evidence, est_cost):
    conn = get_conn()
    conn.execute(
        "INSERT INTO decisions (decided_at, order_id, action, status, reason, evidence, est_cost) "
        "VALUES (?,?,?,?,?,?,?)",
        (datetime.now().isoformat(timespec="seconds"), order_id, action, status,
         reason, evidence, est_cost),
    )
    conn.commit()
    conn.close()


def list_decisions():
    conn = get_conn()
    df = pd.read_sql_query("SELECT * FROM decisions ORDER BY id DESC", conn)
    conn.close()
    return df
