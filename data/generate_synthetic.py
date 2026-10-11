"""Creates supply.db with SYNTHETIC data. Run this once (re-run any time to reset).
Order O-1042 is built to match the demo on your slides."""
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))  # lets us import core.*

import numpy as np
import pandas as pd

from core.db import DB_PATH, get_conn
from core import decisions

rng = np.random.default_rng(42)           # fixed seed -> same data every run
REF_TIME = datetime.now().replace(microsecond=0)
REF_DATE = REF_TIME.replace(hour=0, minute=0, second=0)

# ---------- static network ----------
ports = pd.DataFrame([
    ("P1", "Chennai", 0.55), ("P2", "Mumbai (JNPT)", 0.35),
    ("P3", "Mundra", 0.25), ("P4", "Visakhapatnam", 0.20),
], columns=["port_id", "port_name", "port_congestion"])

routes = pd.DataFrame([
    ("R1", "P1", "Apex Components", 0.90, 6), ("R2", "P1", "Bharat Castings", 0.85, 8),
    ("R3", "P2", "Coastal Electronics", 0.93, 5), ("R4", "P2", "Deccan Alloys", 0.88, 9),
    ("R5", "P3", "Eastern Polymers", 0.95, 4), ("R6", "P4", "Falcon Fasteners", 0.82, 7),
], columns=["route_id", "port_id", "supplier", "supplier_reliability", "base_transit_days"])

# ---------- inventory (SKU-101 has 1.2 days of cover, like the slide) ----------
cover_by_sku = {"SKU-100": 6.0, "SKU-101": 1.2, "SKU-102": 3.5, "SKU-103": 2.0, "SKU-104": 8.0}
consumption = {"SKU-100": 15, "SKU-101": 10, "SKU-102": 20, "SKU-103": 8, "SKU-104": 12}
inventory = pd.DataFrame([
    (sku, round(cover_by_sku[sku] * consumption[sku]), consumption[sku]) for sku in cover_by_sku
], columns=["sku", "stock_units", "daily_consumption"])

alt_units = {sku: int(rng.integers(100, 500)) for sku in cover_by_sku}
alt_units["SKU-101"] = 300
alt_stock = pd.DataFrame(list(alt_units.items()), columns=["sku", "units"])

# ---------- shipments in transit (S-204 is the slide's shipment) ----------
ship_routes = ["R1", "R2", "R3", "R1", "R4", "R5", "R2", "R6", "R3", "R1", "R4", "R5"]
skus = list(cover_by_sku)
shipments = []
for i, route in enumerate(ship_routes):
    sid = f"S-{201 + i}"
    sku = "SKU-101" if sid == "S-204" else skus[i % len(skus)]
    eta = REF_DATE + timedelta(days=2 + (i % 7))
    last_update = REF_TIME - timedelta(hours=float(rng.uniform(1, 3)))
    shipments.append((sid, route, sku, int(rng.integers(100, 600)),
                      eta.date().isoformat(), last_update.isoformat(timespec="seconds")))
shipments = pd.DataFrame(shipments, columns=["shipment_id", "route_id", "sku", "qty", "eta", "last_update"])

# ---------- customer orders ----------
customers = ["Tata Motors", "Bosch India", "Mahindra", "L&T", "TVS", "Ashok Leyland", "Bajaj", "Eicher"]
orders = []
for n in range(1010, 1070):
    oid = f"O-{n}"
    if oid == "O-1042":
        ship, qty, buffer = shipments[shipments.shipment_id == "S-204"].iloc[0], 200, 1
    else:
        ship = shipments.iloc[int(rng.integers(0, len(shipments)))]
        qty, buffer = int(rng.integers(50, 300)), int(rng.choice([0, 1, 1, 2, 3, 4]))
    promised = pd.Timestamp(ship.eta) + timedelta(days=buffer)
    orders.append((oid, str(rng.choice(customers)), ship.sku, qty,
                   promised.date().isoformat(), ship.shipment_id))
orders = pd.DataFrame(orders, columns=["order_id", "customer", "sku", "qty", "promised_date", "shipment_id"])

# ---------- one live disruption: Chennai closed 2 days ----------
events = pd.DataFrame([("E-1", "P1", "Port closure", 2, REF_TIME.isoformat(timespec="seconds"))],
                      columns=["event_id", "port_id", "event_type", "duration_days", "detected_at"])

# ---------- history for training the ML model ----------
n = 4000
hist = pd.DataFrame({
    "ship_date": pd.date_range("2024-01-01", periods=n, freq="5h").strftime("%Y-%m-%d %H:%M"),
    "base_transit_days": rng.integers(3, 13, n),
    "port_congestion": rng.beta(2, 5, n).round(3),
    "event_active": (rng.random(n) < 0.12).astype(int),
    "supplier_reliability": rng.uniform(0.7, 0.98, n).round(3),
    "buffer_days": rng.integers(0, 5, n),
})
hist["event_duration"] = np.where(hist.event_active == 1, rng.integers(1, 6, n), 0)
# Hidden "true" rule the model has to discover (plus noise):
delay = (0.2 + 2.0 * hist.port_congestion + 0.9 * hist.event_duration
         + 3.0 * (1 - hist.supplier_reliability) + rng.normal(0, 0.7, n)).clip(lower=0)
hist["late"] = (delay > hist.buffer_days).astype(int)

# ---------- write everything ----------
conn = get_conn()
for name, df in [("ports", ports), ("routes", routes), ("inventory", inventory),
                 ("alt_stock", alt_stock), ("shipments", shipments), ("orders", orders),
                 ("events", events), ("history", hist)]:
    df.to_sql(name, conn, if_exists="replace", index=False)
pd.DataFrame([("reference_time", REF_TIME.isoformat(timespec="seconds"))],
             columns=["key", "value"]).to_sql("meta", conn, if_exists="replace", index=False)
conn.close()
decisions.init_table()
print(f"Created {DB_PATH} | orders={len(orders)} shipments={len(shipments)} history={len(hist)}")
print(f"Late rate in history: {hist.late.mean():.1%}")
