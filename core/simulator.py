"""What-if simulator: compare today's situation (baseline) with a hypothetical disruption."""
from core import analysis, options


def summarize(net, mdl, delay_map):
    df = analysis.assess(net, delay_map, mdl)
    if df.empty:
        return {"orders_affected": 0, "breaches": 0, "recovery_cost": 0, "unresolved": 0}
    alt = net["alt_stock"].set_index("sku")["units"]
    cost, unresolved = 0.0, 0
    for _, row in df[df.breach].iterrows():
        rec = options.recommend(options.generate_options(row, alt[row["sku"]]))
        if rec is None:
            unresolved += 1
        else:
            cost += rec.cost
    return {"orders_affected": len(df), "breaches": int(df.breach.sum()),
            "recovery_cost": cost, "unresolved": unresolved}


def simulate(net, mdl, port_id, duration_days):
    base_map = analysis.live_delay_map(net)
    sim_map = dict(base_map)
    sim_map[port_id] = max(sim_map.get(port_id, 0), duration_days)
    return summarize(net, mdl, base_map), summarize(net, mdl, sim_map)
