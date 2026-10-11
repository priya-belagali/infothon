"""The planner's screen. Run with:  streamlit run app/streamlit_app.py"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st

from agent import pipeline
from core import analysis, decisions, model, simulator

import subprocess
from core.db import DB_PATH

st.set_page_config(page_title="Supply Chain Disruption Agent", layout="wide")

# On a fresh server there is no database yet, so build the synthetic data once.
if not DB_PATH.exists():
    root = Path(__file__).resolve().parent.parent
    subprocess.run([sys.executable, str(root / "data" / "generate_synthetic.py")], check=True)
decisions.init_table()


@st.cache_resource
def get_model():
    return model.train_model()          # trains once, then reused


mdl = get_model()
net = analysis.load_network()
delay_map = analysis.live_delay_map(net)
port_names = net["ports"].set_index("port_id")["port_name"].to_dict()

# ---------------- sidebar ----------------
st.sidebar.title("Disruption Agent")
page = st.sidebar.radio("Page", ["Dashboard", "Order detail", "What-if simulator",
                                 "Decision log", "Model check"])
stale_demo = st.sidebar.checkbox("Simulate stale data (+9 hours)")
extra_hours = 9 if stale_demo else 0
st.sidebar.markdown("**Live disruptions**")
for pid, d in delay_map.items():
    st.sidebar.warning(f"{port_names[pid]}: closed {d} day(s)")

df = analysis.assess(net, delay_map, mdl, extra_hours)

# ---------------- pages ----------------
if page == "Dashboard":
    st.title("Exposed orders")
    if df.empty:
        st.success("No orders are exposed to the current disruptions.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Orders affected", len(df))
        c2.metric("High risk", int((df.risk_label == "HIGH").sum()))
        c3.metric("Potential delivery breaches", int(df.breach.sum()))
        if df.stale.any():
            st.warning("SAFE MODE - shipment data is stale. Prediction reliability may be reduced; "
                       "verify manually before approving high-impact actions.")
        cols = ["order_id", "customer", "sku", "risk_score", "risk_label", "ml_prob",
                "delay_days", "buffer_days", "cover_days", "breach", "port_name"]
        st.dataframe(df[cols], use_container_width=True, hide_index=True)
        st.caption("Risk score is rule-based (not a calibrated probability). ml_prob is the XGBoost late-shipment estimate.")

elif page == "Order detail":
    st.title("Why is this order at risk?")
    if df.empty:
        st.info("Nothing to show - no exposed orders.")
        st.stop()
    default = df.index[df.order_id == "O-1042"].tolist()
    order_id = st.selectbox("Order", df.order_id.tolist(), index=default[0] if default else 0)
    res = pipeline.run_agent(net, mdl, order_id, delay_map, extra_hours)
    row = res["row"]

    a, b, c = st.columns(3)
    a.metric("Risk score", f"{row.risk_score}/100", row.risk_label, delta_color="off")
    b.metric("Confidence", res["confidence"])
    c.metric("ML late probability", f"{row.ml_prob:.0%}")

    if res["stale"]:
        st.warning(f"DATA QUALITY WARNING: shipment information is {res['hours_old']:.0f} hours old. "
                   "Refresh the data or verify the shipment manually.")

    st.subheader("Evidence")
    for line in res["evidence"]:
        st.write("- " + line)
    st.caption("Top model factors: " + ", ".join(f"{n} ({v:+.2f})" for n, v in res["ml_factors"]))
    if res["llm_summary"]:
        st.info(res["llm_summary"])
    with st.expander("Agent steps (tool calls)"):
        for s in res["steps"]:
            st.code(s)

    st.subheader("Recovery options")
    st.table(pd.DataFrame([o.to_dict() for o in res["options"]]))
    rec = res["recommended"]
    if rec is None:
        st.error("No feasible plan - escalate to a human planner.")
    else:
        st.success(f"Recommended: {rec.name} (Rs {rec.cost:,.0f}, residual delay {rec.delay_days:g} day(s))")
        names = [o.name for o in res["options"]]
        chosen_name = st.selectbox("Action to approve (default = recommendation)", names,
                                   index=names.index(rec.name))
        chosen = next(o for o in res["options"] if o.name == chosen_name)
        verified = True
        if res["stale"]:
            verified = st.checkbox("I have manually verified the shipment data")
        evidence_txt = (f"Risk {row.risk_score}/100 · {row.shipment_id} +{row.delay_days}d · "
                        f"cover {row.cover_days}d")
        c1, c2 = st.columns(2)
        if c1.button("APPROVE", disabled=not verified):
            decisions.log_decision(order_id, chosen.name, "APPROVED by planner",
                                   f"{chosen.name}: est. Rs {chosen.cost:,.0f}, residual delay {chosen.delay_days:g}d",
                                   evidence_txt, chosen.cost)
            st.success("Decision logged. (No operational system was changed - this is a demo.)")
        if c2.button("REJECT"):
            decisions.log_decision(order_id, chosen.name, "REJECTED by planner",
                                   "Planner rejected the recommendation", evidence_txt, chosen.cost)
            st.info("Rejection logged.")

elif page == "What-if simulator":
    st.title("What-if disruption simulator")
    c1, c2 = st.columns(2)
    port_id = c1.selectbox("Disruption: port closure at", list(port_names), format_func=port_names.get)
    days = c2.slider("Duration (days)", 1, 7, 3)
    if st.button("SIMULATE IMPACT"):
        base, sim = simulator.simulate(net, mdl, port_id, days)
        table = pd.DataFrame({
            "Baseline": [base["orders_affected"], base["breaches"], f"Rs {base['recovery_cost']:,.0f}"],
            "Simulated": [sim["orders_affected"], sim["breaches"], f"Rs {sim['recovery_cost']:,.0f}"],
        }, index=["Orders affected", "Potential delivery breaches", "Est. recovery cost"])
        st.table(table)
        if sim["unresolved"]:
            st.error(f"{sim['unresolved']} breaching order(s) have no feasible plan - escalate to a human.")
        st.caption("Computed from the synthetic demo data - not real-world results.")

elif page == "Decision log":
    st.title("Decision log")
    log = decisions.list_decisions()
    if log.empty:
        st.info("No decisions yet. Approve or reject something on the Order detail page.")
    else:
        st.dataframe(log, use_container_width=True, hide_index=True)

else:
    st.title("Model check: rules vs ML")
    m = mdl.metrics
    st.write(f"Model: **{m['kind']}** · trained on the oldest {m['train_rows']} shipments, "
             f"tested on the newest {m['test_rows']} (time-based split, no future leakage).")
    st.table(pd.DataFrame({"Precision": [m["rule"]["precision"], m["ml"]["precision"]],
                           "Recall": [m["rule"]["recall"], m["ml"]["recall"]]},
                          index=["Simple rule", "ML model"]).round(3))
    st.caption("Synthetic data: these scores prove the pipeline works, not real-world accuracy.")
