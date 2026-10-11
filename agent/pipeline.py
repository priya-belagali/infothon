"""The 'agent': a chain of tools (sense -> predict -> trace -> plan). Every number comes from a tool.
Optional: if ANTHROPIC_API_KEY is set, an LLM can rewrite the evidence as a friendly summary."""
import os

from core import analysis, options, graph


def run_agent(net, mdl, order_id, delay_map, extra_stale_hours=0.0):
    steps = []

    # 1) SENSE - which disruptions are live?
    ports = net["ports"].set_index("port_id")["port_name"]
    steps.append("SENSE: " + ", ".join(f"{ports[p]} disrupted for {d} day(s)" for p, d in delay_map.items()))

    # 2) PREDICT - score the order
    df = analysis.assess(net, delay_map, mdl, extra_stale_hours)
    row = df[df.order_id == order_id].iloc[0]
    steps.append(f"PREDICT: rule risk {row.risk_score}/100 ({row.risk_label}), ML late probability {row.ml_prob:.0%}")

    # 3) TRACE - show the dependency chain
    G = graph.build_graph(net)
    chain = graph.trace_path(G, row.port_id, order_id)
    steps.append("TRACE: " + " -> ".join(chain))

    # 4) PLAN - options and recommendation
    alt = net["alt_stock"].set_index("sku")["units"]
    opts = options.generate_options(row, alt[row.sku])
    rec = options.recommend(opts)
    steps.append("PLAN: recommended = " + (rec.name if rec else "NONE - escalate to a human"))

    # evidence in plain language (the "why")
    evidence = [
        f"{row.port_name} disruption delays shipment {row.shipment_id} by {row.delay_days} day(s).",
        f"{row.sku} has only {row.cover_days} days of stock cover.",
        f"Projected delay ({row.delay_days}d) {'exceeds' if row.delay_days > row.buffer_days else 'is within'} "
        f"the delivery buffer ({row.buffer_days}d).",
    ]

    # confidence: lowered when data is stale
    if row.stale:
        confidence = "LOW"
    elif row.risk_label == "HIGH" and row.ml_prob >= 0.5:
        confidence = "HIGH"
    else:
        confidence = "MEDIUM"

    result = {
        "row": row, "steps": steps, "evidence": evidence, "confidence": confidence,
        "options": opts, "recommended": rec, "stale": bool(row.stale), "hours_old": row.hours_old,
        "ml_factors": mdl.explain(row.to_dict()),
    }
    result["llm_summary"] = llm_summary(result)
    return result


def llm_summary(result):
    """Optional. Returns None if no API key / package / network - the app works without it."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        return None
    try:
        import anthropic
        prompt = ("You are a supply-chain planning assistant. In 3 sentences, explain this risk to a planner. "
                  "Use ONLY these facts:\n" + "\n".join(result["evidence"]) +
                  f"\nRecommended action: {result['recommended'].name if result['recommended'] else 'escalate'}")
        client = anthropic.Anthropic()
        msg = client.messages.create(model="claude-sonnet-5-5", max_tokens=300,
                                     messages=[{"role": "user", "content": prompt}])
        return msg.content[0].text
    except Exception:
        return None
