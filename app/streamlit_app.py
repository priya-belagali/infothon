"""The planner's screen. Run with:  python -m streamlit run app/streamlit_app.py"""
import subprocess
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st

from agent import pipeline
from core import analysis, decisions, model, simulator, ui, weather
from core.db import DB_PATH

st.set_page_config(page_title="Supply Chain Disruption Agent", layout="wide")
ui.inject_css()
SHIP_GIF = Path(__file__).resolve().parent / "Container_ship_transparent.gif"

# On a fresh server there is no database yet, so build the synthetic data once.
if not DB_PATH.exists():
    root = Path(__file__).resolve().parent.parent
    subprocess.run([sys.executable, str(root / "data" / "generate_synthetic.py")], check=True)
decisions.init_table()


@st.cache_resource
def get_model():
    return model.train_model()          # trains once, then reused


@st.cache_data(ttl=1800, show_spinner="Reading live weather...")
def get_weather():
    return weather.load_weather()          # cached 30 minutes so we don't hammer the API


mdl = get_model()
net = analysis.load_network()
port_names = net["ports"].set_index("port_id")["port_name"].to_dict()

# ---------------- sidebar ----------------
P_DASH, P_ORDER, P_SIM = "Dashboard", "Order detail", "What-if simulator"
P_WEATHER, P_LOG, P_MODEL = "Weather", "Decision log", "Model check"

st.sidebar.markdown("## Disruption Agent")
st.sidebar.caption("Predict · Trace · Approve")

# ---- dark / light mode switch ----
if "dark_mode" not in st.session_state:
    st.session_state["dark_mode"] = st.get_option("theme.base") != "light"
_toggle = getattr(st.sidebar, "toggle", st.sidebar.checkbox)
dark_mode = _toggle("Dark mode", key="dark_mode")
if ui.theme_needs_change(dark_mode) and st.session_state.get("_theme_tried") != dark_mode:
    st.session_state["_theme_tried"] = dark_mode      # try once per switch, never loop
    if ui.apply_theme(dark_mode):
        (getattr(st, "rerun", None) or st.experimental_rerun)()
if ui.theme_needs_change(dark_mode):
    st.sidebar.caption("Theme switch isn't available in this version - use the menu (three dots, top right) > Settings > Theme.")

page = st.sidebar.radio("Go to", [P_DASH, P_ORDER, P_SIM, P_WEATHER, P_LOG, P_MODEL])
stale_demo = st.sidebar.checkbox("Simulate stale data (+9 hours)")
extra_hours = 9 if stale_demo else 0

st.sidebar.markdown("**Disruption sources**")
use_db_event = st.sidebar.checkbox("Port closure from database (demo)", value=True)
use_weather = st.sidebar.checkbox("Live weather (Open-Meteo)")
demo_storm = st.sidebar.checkbox("Demo: inject a storm at Chennai")

db_map = analysis.live_delay_map(net) if use_db_event else {}
raw_weather = get_weather() if use_weather else {}
weather_failed = [pid for pid, v in raw_weather.items() if v.get("error")]
weather_info = raw_weather
if demo_storm:
    weather_info = weather.with_demo_storm(weather_info, "P1", 3)
weather_map = weather.to_delay_map(weather_info)
delay_map = weather.merge_delays(db_map, weather_map)

st.sidebar.markdown("**Live disruptions**")
if not delay_map:
    st.sidebar.success("None right now")
for pid, d in db_map.items():
    st.sidebar.error(f"{port_names[pid]}: port closed {d} day(s)")
for pid, d in weather_map.items():
    st.sidebar.warning(f"{port_names[pid]}: severe weather {d} day(s)")



def toast_on_change(key, value, on_true, on_false=None):
    """Pop a toast only when a sidebar switch CHANGES (not on every rerun)."""
    prev = st.session_state.get(f"_prev_{key}")
    if prev is not None and prev != value:
        msg = on_true if value else on_false
        if msg:
            st.toast(msg)
    st.session_state[f"_prev_{key}"] = value


toast_on_change("stale", stale_demo,
                "Safe mode ON: shipment data is 9 hours old.", "Shipment data is fresh again.")
toast_on_change("storm", demo_storm, "Demo storm injected at Chennai.", "Demo storm cleared.")
toast_on_change("dbevent", use_db_event,
                "Database port closure included.", "Database port closure removed.")
toast_on_change("weather", use_weather,
                f"Weather unavailable for {len(weather_failed)} port(s). Check your internet connection."
                if weather_failed else "Live weather loaded.",
                "Live weather switched off.")

df = analysis.assess(net, delay_map, mdl, extra_hours)

# ---------------- pages ----------------
if page == P_DASH:
    ui.hero("Supply Chain Disruption Agent",
            "Predict the delay. Trace the impact. Choose the next move.",
            tags=["SYNTHETIC DEMO DATA", "HUMAN-APPROVED ACTIONS ONLY"])

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Active disruptions", len(delay_map))
    k2.metric("Orders affected", len(df))
    k3.metric("High risk", int((df.risk_label == "HIGH").sum()) if not df.empty else 0)
    k4.metric("Potential breaches", int(df.breach.sum()) if not df.empty else 0)

    if df.empty:
        st.success("No orders are exposed to the current disruptions.")
    else:
        if df.stale.any():
            st.warning("SAFE MODE - shipment data is stale. Prediction reliability may be reduced; "
                       "verify manually before approving high-impact actions.")
        left, right = st.columns([3, 2])
        with left:
            st.subheader("Highest-risk orders")
            st.altair_chart(ui.risk_bar_chart(df))
        with right:
            st.subheader("Risk mix")
            st.altair_chart(ui.risk_donut(df))

    st.subheader("Supply routes")
    try:
        st.pydeck_chart(ui.route_map(net["ports"], net["routes"], net["shipments"],
                                     weather.PORT_COORDS, delay_map))
        st.caption("Arcs run from each port to the factory. Red = disrupted, green = operating. "
                   "Thicker arc = more shipments in transit. Hover for details.")
    except Exception:
        map_df = ui.port_map_df(net["ports"], weather.PORT_COORDS, delay_map)
        try:
            st.map(map_df, latitude="lat", longitude="lon", color="color", size="size")
        except Exception:
            st.map(map_df[["lat", "lon"]])
        st.caption("Red = disrupted port, green = operating normally.")

    if not df.empty:
        st.subheader("All exposed orders")
        view = df.copy()
        view["risk_label"] = view["risk_label"]
        view["ml_prob"] = (view["ml_prob"] * 100).round(0)
        view["breach"] = view["breach"].map(lambda b: "Breach" if b else "OK")
        cols = ["order_id", "customer", "sku", "risk_score", "risk_label", "ml_prob",
                "delay_days", "buffer_days", "cover_days", "breach", "port_name"]
        st.dataframe(
            view[cols], use_container_width=True, hide_index=True,
            column_config={
                "risk_score": st.column_config.ProgressColumn("Risk score", min_value=0, max_value=100, format="%d"),
                "ml_prob": st.column_config.ProgressColumn("ML late prob (%)", min_value=0, max_value=100, format="%d"),
            })
        st.caption("Risk score is rule-based (not a calibrated probability). "
                   "ML late prob is the model's late-shipment estimate.")

elif page == P_ORDER:
    if df.empty:
        ui.hero("Why is this order at risk?", "Evidence first, action second.")
        st.info("Nothing to show - no exposed orders.")
        st.stop()
    default = df.index[df.order_id == "O-1042"].tolist()
    order_id = st.selectbox("Choose an order", df.order_id.tolist(), index=default[0] if default else 0)
    res = pipeline.run_agent(net, mdl, order_id, delay_map, extra_hours)
    row, rec = res["row"], res["recommended"]

    ui.hero(f"Order {order_id}", f"{row.customer} · {row.sku} · quantity {row.qty}",
            tags=[f"{row.risk_label} RISK",
                  f"CONFIDENCE: {res['confidence']}"])

    st.markdown(ui.flow_html([
        ("1", "Disruption", f"{row.port_name}: {row.delay_days} day(s)", "bad"),
        ("2", "Shipment delay", f"{row.shipment_id} +{row.delay_days}d", "bad"),
        ("3", "Stock cover", f"{row.sku}: {row.cover_days}d",
         "bad" if row.cover_days < row.delay_days else ""),
        ("4", "Order at risk", f"Buffer {row.buffer_days}d", "bad" if row.breach else ""),
        ("5", "Recovery", rec.name if rec else "Escalate to human", "good"),
    ]), unsafe_allow_html=True)

    if res["stale"]:
        st.warning(f"DATA QUALITY WARNING: shipment information is {res['hours_old']:.0f} hours old. "
                   "Refresh the data or verify the shipment manually.")

    tab_over, tab_evid, tab_opts, tab_dec = st.tabs(["Overview", "Evidence", "Options", "Decision"])

    with tab_over:
        g1, g2, g3 = st.columns(3)
        g1.markdown(ui.gauge_html("Rule-based risk", f"{row.risk_score}/100", row.risk_score / 100,
                                  ui.RISK_ZONES, "Ranks orders for review"), unsafe_allow_html=True)
        g2.markdown(ui.gauge_html("ML late probability", f"{row.ml_prob:.0%}", float(row.ml_prob),
                                  ui.PROB_ZONES, "Model estimate, not calibrated"), unsafe_allow_html=True)
        with g3:
            st.metric("Confidence", res["confidence"])
            if rec:
                detail = f"Est. cost Rs {rec.cost:,.0f} · residual delay {rec.delay_days:g} day(s)"
                st.markdown(ui.rec_card_html("Recommended action", rec.name, detail), unsafe_allow_html=True)
            else:
                st.markdown(ui.rec_card_html("Recommended action", "Escalate to a human planner",
                                             "No feasible recovery plan was found."), unsafe_allow_html=True)

    with tab_evid:
        e1, e2 = st.columns([3, 2])
        with e1:
            st.subheader("Evidence")
            st.markdown(ui.evidence_html(res["evidence"]), unsafe_allow_html=True)
            if res["llm_summary"]:
                st.info(res["llm_summary"])
            with st.expander("Agent steps (tool calls)"):
                for step in res["steps"]:
                    st.code(step)
        with e2:
            st.subheader("What drives the prediction")
            st.altair_chart(ui.factor_chart(res["ml_factors"]))

    with tab_opts:
        st.subheader("Recovery options")
        cols = st.columns(len(res["options"]))
        for col, o in zip(cols, res["options"]):
            col.markdown(ui.option_card_html(o, rec is not None and o.name == rec.name),
                         unsafe_allow_html=True)
        if rec is None:
            st.error("No feasible plan - escalate to a human planner.")

    with tab_dec:
        st.subheader("Planner decision")
        if rec is None:
            st.error("No feasible plan - escalate to a human planner.")
        else:
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
            if c1.button("APPROVE", disabled=not verified, use_container_width=True):
                decisions.log_decision(order_id, chosen.name, "APPROVED by planner",
                                       f"{chosen.name}: est. Rs {chosen.cost:,.0f}, residual delay {chosen.delay_days:g}d",
                                       evidence_txt, chosen.cost)
                st.toast(f"Decision logged: {chosen.name} for {order_id}")
                st.success("Decision logged. (No operational system was changed - this is a demo.)")
                if SHIP_GIF.exists():
                    st.markdown(ui.ship_html(SHIP_GIF, "Approved - cargo is on the move"),
                                unsafe_allow_html=True)
            if c2.button("REJECT", use_container_width=True):
                decisions.log_decision(order_id, chosen.name, "REJECTED by planner",
                                       "Planner rejected the recommendation", evidence_txt, chosen.cost)
                st.toast(f"Rejection logged for {order_id}")
                st.info("Rejection logged.")

elif page == P_SIM:
    ui.hero("What-if disruption simulator", "Test a disruption before committing to a recovery plan.",
            tags=["ILLUSTRATIVE - COMPUTED FROM SYNTHETIC DATA"])
    c1, c2 = st.columns(2)
    port_id = c1.selectbox("Disruption: port closure at", list(port_names), format_func=port_names.get)
    days = c2.slider("Duration (days)", 1, 7, 3)
    if st.button("SIMULATE IMPACT"):
        base, sim = simulator.simulate(net, mdl, port_id, days, base_map=delay_map)
        m1, m2, m3 = st.columns(3)
        m1.metric("Orders affected", sim["orders_affected"],
                  delta=f"{sim['orders_affected'] - base['orders_affected']:+d}", delta_color="inverse")
        m2.metric("Potential delivery breaches", sim["breaches"],
                  delta=f"{sim['breaches'] - base['breaches']:+d}", delta_color="inverse")
        m3.metric("Est. recovery cost", f"Rs {sim['recovery_cost']:,.0f}",
                  delta=f"Rs {sim['recovery_cost'] - base['recovery_cost']:+,.0f}", delta_color="inverse")
        st.toast("Simulation complete")
        st.altair_chart(ui.sim_chart(base, sim))
        if sim["unresolved"]:
            st.error(f"{sim['unresolved']} breaching order(s) have no feasible plan - escalate to a human.")
        st.caption("Computed from the synthetic demo data - not real-world results.")

elif page == P_WEATHER:
    ui.hero("Port weather", "Severe weather becomes a disruption the agent can reason about.")
    if not weather_info:
        st.info("Tick 'Live weather (Open-Meteo)' in the sidebar to read real forecasts, "
                "or 'Demo: inject a storm' to simulate one.")
    else:
        chart = ui.gust_chart(weather_info, port_names, weather.GUST_LIMIT_KMH)
        if chart is not None:
            st.altair_chart(chart)
            st.caption(f"Dashed line = severe-weather limit ({weather.GUST_LIMIT_KMH} km/h gusts).")
        st.dataframe(weather.weather_table(weather_info, port_names),
                     use_container_width=True, hide_index=True)
        st.caption(f"A day is 'severe' if gusts >= {weather.GUST_LIMIT_KMH} km/h or rain >= "
                   f"{weather.RAIN_LIMIT_MM} mm. Severe days in the next {weather.FORECAST_DAYS} days "
                   "become disruption days. These thresholds are assumptions, not validated.")

elif page == P_LOG:
    ui.hero("Decision log", "Every approval and rejection, with the evidence behind it.")
    log = decisions.list_decisions()
    if log.empty:
        st.info("No decisions yet. Approve or reject something on the Order detail page.")
    else:
        approved = log["status"].str.contains("APPROVED")
        m1, m2, m3 = st.columns(3)
        m1.metric("Decisions logged", len(log))
        m2.metric("Approved", int(approved.sum()))
        m3.metric("Approved spend", f"Rs {log.loc[approved, 'est_cost'].sum():,.0f}")
        st.dataframe(log, use_container_width=True, hide_index=True)

else:
    ui.hero("Model check: rules vs ML", "Prove value before claiming accuracy.",
            tags=["TIME-BASED SPLIT", "NO FUTURE LEAKAGE"])
    m = mdl.metrics
    st.write(f"Model: **{m['kind']}** · trained on the oldest {m['train_rows']} shipments, "
             f"tested on the newest {m['test_rows']}.")
    st.altair_chart(ui.model_chart(m))
    st.table(pd.DataFrame({"Precision": [m["rule"]["precision"], m["ml"]["precision"]],
                           "Recall": [m["rule"]["recall"], m["ml"]["recall"]]},
                          index=["Simple rule", "ML model"]).round(3))
    st.caption("Synthetic data: these scores prove the pipeline works, not real-world accuracy.")
