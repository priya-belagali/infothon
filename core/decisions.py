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


def get_order_decision(order_id):
    """Returns the most recent decision record for an order, or None if not decided yet."""
    conn = get_conn()
    cur = conn.execute("SELECT * FROM decisions WHERE order_id = ? ORDER BY id DESC LIMIT 1", (order_id,))
    row = cur.fetchone()
    if row is not None:
        cols = [d[0] for d in cur.description]
        res = dict(zip(cols, row))
        conn.close()
        return res
    conn.close()
    return None


def clear_decisions(order_id=None):
    """Clear decisions log (optionally for a specific order). Useful for testing/demo reset."""
    conn = get_conn()
    if order_id:
        conn.execute("DELETE FROM decisions WHERE order_id = ?", (order_id,))
    else:
        conn.execute("DELETE FROM decisions")
    conn.commit()
    conn.close()
