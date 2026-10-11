"""Glue: load the data, trace exposure, score every exposed order."""
import pandas as pd

from core import graph, risk
from core.db import get_conn

STALE_HOURS = 6   # shipment data older than this triggers Safe Mode


def load_network():
    conn = get_conn()
    net = {t: pd.read_sql_query(f"SELECT * FROM {t}", conn)
           for t in ["ports", "routes", "shipments", "inventory", "orders", "alt_stock", "events"]}
    meta = pd.read_sql_query("SELECT * FROM meta", conn)
    conn.close()
    net["ref_time"] = pd.to_datetime(meta.loc[meta.key == "reference_time", "value"].iloc[0])
    net["shipments"]["eta"] = pd.to_datetime(net["shipments"]["eta"])
    net["shipments"]["last_update"] = pd.to_datetime(net["shipments"]["last_update"])
    net["orders"]["promised_date"] = pd.to_datetime(net["orders"]["promised_date"])
    return net


def live_delay_map(net):
    """{port_id: delay_days} for the disruptions currently in the events table."""
    out = {}
    for _, e in net["events"].iterrows():
        out[e.port_id] = max(out.get(e.port_id, 0), int(e.duration_days))
    return out


def assess(net, delay_by_port, mdl=None, extra_stale_hours=0.0):
    """One row per exposed order, with risk score, breach flag and (optionally) ML probability."""
    G = graph.build_graph(net)
    orders = net["orders"].set_index("order_id")
    ships = net["shipments"].set_index("shipment_id")
    routes = net["routes"].set_index("route_id")
    ports = net["ports"].set_index("port_id")
    inv = net["inventory"].set_index("sku")

    rows = []
    for port_id, delay in delay_by_port.items():
        for oid in graph.exposed_orders(G, port_id):
            o = orders.loc[oid]
            s = ships.loc[o.shipment_id]
            r = routes.loc[s.route_id]
            cover = inv.loc[o.sku, "stock_units"] / inv.loc[o.sku, "daily_consumption"]
            buffer = (o.promised_date - s.eta).days
            hours_old = (net["ref_time"] - s.last_update).total_seconds() / 3600 + extra_stale_hours
            score = risk.rule_score(delay, buffer, cover)
            rows.append({
                "order_id": oid, "customer": o.customer, "sku": o.sku, "qty": int(o.qty),
                "promised_date": o.promised_date.date(), "shipment_id": o.shipment_id,
                "route_id": s.route_id, "port_id": port_id, "port_name": ports.loc[port_id, "port_name"],
                "delay_days": delay, "buffer_days": buffer, "cover_days": round(cover, 1),
                "risk_score": score, "risk_label": risk.risk_label(score),
                "breach": risk.is_breach(delay, buffer, cover),
                "hours_old": round(hours_old, 1), "stale": hours_old > STALE_HOURS,
                # model inputs
                "base_transit_days": r.base_transit_days, "port_congestion": ports.loc[port_id, "port_congestion"],
                "event_active": 1, "event_duration": delay, "supplier_reliability": r.supplier_reliability,
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    if mdl is not None:
        df["ml_prob"] = mdl.predict_proba(df).round(2)
    return df.sort_values("risk_score", ascending=False).reset_index(drop=True)
